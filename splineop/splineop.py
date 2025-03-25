import numpy as np
from numba import njit, objmode
from numba.experimental import jitclass
from numba import int64, float64, int64, float64
from scipy import interpolate
from scipy.stats import dirichlet
import matplotlib.pyplot as plt
from splineop.costs import *
import pdb
from timeit import default_timer as timer 
from sklearn.linear_model import LinearRegression


splineop_spec_Pen = [("cost", costPenalized.class_type.instance_type)]
@jitclass(splineop_spec_Pen)
class splineOPPenalized(object):
    n_points: int64
    n_states: int64
    states: float64[:]
    initial_speeds: float64[:]
    bkps: float64[:]
    knots: float64[:]
    state_idx_sequence: int64[:]
    time_path_mat: int64[:, :]
    soc: float64[:, :]
    state_path_mat: int64[:, :]
    speed_path_mat: float64[:, :]
    execution_time: float64
    t_start : float64

    def __init__(self, cost_fn):
        self.cost = cost_fn

    def fit(
        self,
        signal: np.ndarray,
        states: np.ndarray,
        initial_speeds: np.ndarray,
        normalized: bool,
    ):
        self.n_points = signal.shape[0]
        self.n_states = states.shape[-1]
        self.states = states  # np.array([_ for _ in set(states)], dtype=np.float64)
        self.initial_speeds = initial_speeds  # np.array([_ for _ in set(initial_speeds)], dtype=np.float64)
        self.cost.fit(signal, states, initial_speeds, normalized)

    def backtrack_solution(self) -> tuple[np.ndarray, np.ndarray]:
        bkps = np.empty(shape=0, dtype=np.float64)
        state_idx_sequence = np.array([int(np.argmin(self.soc[-1]))], dtype=np.int64)
        t = self.soc.shape[0] - 1
        while t > 0:
            bkps = np.append(bkps, t)
            state_idx_sequence = np.hstack(
                (
                    np.array(
                        [self.state_path_mat[t, state_idx_sequence[0]]], dtype=np.int64
                    ),
                    state_idx_sequence,
                )
            )
            t = self.time_path_mat[t, state_idx_sequence[1]]

        bkps = np.flip(bkps)
        bkps = np.hstack((np.array([0], dtype=np.int64), bkps))
        # Include extremes so that we can reconstruct a scipy.interpolate.Polynomial
        # Directly from this attribute
        self.knots = bkps
        # "Real" changepoints are the ones that are useful for the user
        # Exlucding endpoints
        self.bkps = bkps[1:-1]
        self.state_idx_sequence = state_idx_sequence

        # if self.cost.normalized:
        #    self.bkps = self.bkps / self.n_points
        #    real_bkps = real_bkps / self.n_points

    def predict(self, penalty=0):
        # Case with change points
        
        self.soc = np.empty(shape=(self.n_points + 1, self.n_states), dtype=np.float64)
        self.soc[0, :] = float(0)
        self.time_path_mat = np.empty(
            shape=(self.n_points + 1, self.n_states), dtype=np.int64
        )
        self.state_path_mat = np.empty(
            shape=(self.n_points + 1, self.n_states), dtype=np.int64
        )
        self.speed_path_mat = np.empty(
            shape=(self.n_points + 1, self.n_states), dtype=np.float64
        )
        with objmode(t_start='float64'):
            t_start = timer()
        for end in range(1, self.n_points + 1):
            for p_end_idx in range(self.n_states):
                (
                    self.soc[end, p_end_idx],
                    self.speed_path_mat[end, p_end_idx],
                    self.state_path_mat[end, p_end_idx],
                    self.time_path_mat[end, p_end_idx],
                    opt_start_speed,
                ) = self.cost.compute_optimal_cost(
                    end=end,
                    p_end_idx=p_end_idx,
                    speed_matrix=self.speed_path_mat,
                    initial_speeds=self.initial_speeds,
                    soc=self.soc,
                    penalty=penalty,
                )
                # Save the optimal starting speed to avoid future calculations
                # Proabbly should just remove this to avoid the IF evaluation T*N times ¯\(o.o)/¯
                if self.time_path_mat[end, p_end_idx] == 0:
                    self.speed_path_mat[0, self.state_path_mat[end, p_end_idx]] = (
                        np.float64(opt_start_speed)
                    )
        with objmode(t_end='float64'):
            t_end = timer()
        self.execution_time = t_end - t_start
        return self.backtrack_solution()


splineop_spec_Constrained = [("cost", costConstrained.class_type.instance_type)]
@jitclass(splineop_spec_Constrained)
class splineOPConstrained(object):
    """ A class that allows to solve the splineOP problem with a fixed number of breaks.

 
    Methods:
    __init__
    fit
    predict
    backtrack_solution
    backtrack_specific

    """
    n_points: int64
    n_states: int64
    states: float64[:, :, :]
    initial_speeds: float64[:,:]
    bkps: int64[:]
    knots: int64[:]
    state_idx_sequence: int64[:]
    time_path_mat: int64[:, :, :]
    soc: float64[:, :,:]
    state_path_mat: int64[:, :, :]
    speed_path_mat: float64[:, :, :, :]
    execution_time: float64
    execution_time_k : float64[:]
    ndims : int64

    def __init__(self, cost_fn):
        """ Constructor for the class.
        
        Arguments:
        cost_fn (splineop.costs.costConstrained)
    
        """
        self.cost = cost_fn

    def fit(
        self,
        signal: np.ndarray,
        states: np.ndarray,
        initial_speeds: np.ndarray,
        normalized: bool,
    ):
        """
        Stores the attributes  and computes the sums needed
        for solving each error in O(1).

        Arguments:
        signal (numpy.ndarray): The input signal.
        states (numpy.ndarray): The states of the system.
        initial_speeds (numpy.ndarray): The initial speeds of the system.
        normalized (bool): (Deprecated, but need to completely remove) Whether the data is normalized. 
        """
        self.n_points = signal.shape[0]
        self.n_states = states.shape[1]
        self.states = states  # np.array([_ for _ in set(states)], dtype=np.float64)
        self.initial_speeds = initial_speeds  # np.array([_ for _ in set(initial_speeds)], dtype=np.float64)
        self.cost.fit(signal, states, initial_speeds, normalized)
        self.ndims = signal.shape[1]

    def predict(self, K):
        """
        Compute the optimal partition for K change-points.

        Arguments:
        K (int): Number of changepoints desired by the user, implying K+1 segments.
        
        Internally we think the procedure in terms of nb of segments
        therefore we need dimension K+2 because we index 0 as a dummy
        and then have indexes 1 through K+1, representing the segments
        Since Python is index-0, we use dimension K+2
        That is: Over axis of K, 0 is dummy, 1 is 1 segments (no change), 
        2 is 2 segments (1 change),...,K+1 is K+1 segments (K changes)

        The fist dimension of SOC is used as a dummy with all 0s
        The first dimensions of the others is a dummy never used
        but helps in terms of clarity with the indexing.
        """
        self.soc = np.ones(
            shape=(K + 2, self.n_points + 1, self.n_states), dtype=np.float64
                    )
        self.soc = self.soc * np.inf
        self.soc[0] = float(0.)  # Dummy
        self.soc[1] = np.inf # Puts infinite weight to segments w/o change 
                             # to avoid having sthing like [0..2][3...T] 
                             # Avoiding a very short first segment for sure. 
        self.time_path_mat = np.empty(
            shape=(K + 2, self.n_points + 1, self.n_states), dtype=np.int64
        )
        self.state_path_mat = np.empty(
            shape=(K + 2, self.n_points + 1, self.n_states), dtype=np.int64
        )
        self.speed_path_mat = np.empty(
            shape=(K + 2, self.n_points + 1, self.n_states, self.ndims), dtype=np.float64
        )
        self.execution_time_k= np.zeros(shape=(K+2), dtype=np.float64)
        
        for k in range(1, K + 2): # nb of segments
            with objmode(t_start_k='float64'):
                t_start_k = timer()
            for end in range(k, self.n_points + 1): # nb of points seen
                #pdb.set_trace()
                for p_end_idx in range(self.n_states): # each state
                    (
                        self.soc[k, end, p_end_idx],
                        self.state_path_mat[k, end, p_end_idx],
                        self.time_path_mat[k, end, p_end_idx],
                        self.speed_path_mat[k, end, p_end_idx],
                    ) = self.cost.compute_optimal_cost(
                        end=end,
                        p_end_idx=p_end_idx,
                        speed_matrix=self.speed_path_mat[k - 1],
                        initial_speeds=self.initial_speeds,
                        soc=self.soc[k - 1],
                        k=k,
                    )
            with objmode(t_end_k='float64'):
                # Escape Numba one moment to register the time taken
                t_end_k = timer()
            self.execution_time_k[k] = t_end_k - t_start_k
        self.execution_time = np.sum(self.execution_time_k)
        self.backtrack_solution()

    def backtrack_solution(self) -> tuple[np.ndarray, np.ndarray]:
        """Finds the sequence of optimal time changes and their states, for the default K."""
        
        K = self.soc.shape[0]
        t = self.soc.shape[1] - 1
        bkps = np.array([t], dtype=np.int64)
        state_idx_sequence = np.array(
            [int(np.argmin(self.soc[K-1, -1]))], dtype=np.int64
        )

        for k in range(K - 1, 0, -1):  # 0 is not included
            previous_cp = np.array(
                [self.time_path_mat[k, bkps[0], state_idx_sequence[0]]]
            )
            previous_state = np.array(
                [self.state_path_mat[k, bkps[0], state_idx_sequence[0]]]
            )

            bkps = np.concat((previous_cp, bkps))
            state_idx_sequence = np.concat((previous_state, state_idx_sequence))
        self.knots = bkps
        self.bkps = bkps[1:-1]
        self.state_idx_sequence = state_idx_sequence

    def backtrack_specific(self, K):
        """
        Finds the sequence of optimal time changes and states, for a given K < original K.
        
        Since for computing the original K we need to fill the matrices for all K,
        we can trace the solutions of lower number of segments from the same execution.
        """

        # K+2 is the last index over the K-axis 
        assert K <= self.soc.shape[0]-2
        K = K+2 # I do +2 here, and -1 in the def of state_idx_seq below  
                # so that the code is resembles more to the "normal" backtrack
        
        # Get the last time and last position
        t = self.soc.shape[1] - 1 # last item's index
        bkps = np.array([t], dtype=np.int64) 
        state_idx_sequence = np.array(
            [int(np.argmin(self.soc[K-1, -1]))], dtype=np.int64
        )
        # Iterate over the previous changes to get the time and state
        for k in range(K - 1, 0, -1):  # 0 is not included
            previous_cp = np.array(
                [self.time_path_mat[k, bkps[0], state_idx_sequence[0]]]
            )
            previous_state = np.array(
                [self.state_path_mat[k, bkps[0], state_idx_sequence[0]]]
            )

            bkps = np.concat((previous_cp, bkps))
            state_idx_sequence = np.concat((previous_state, state_idx_sequence))
        self.knots = bkps
        self.bkps = bkps[1:-1]
        self.state_idx_sequence = state_idx_sequence


def plot_pw_results(
    polynomial,
    model,
    show_predicted: bool = True,
    legend: bool = False,
    show_states: bool = False,
):
    """
    Plots the reconstructed pw-quadratic polynomial and the observed datapoints.
    Assumes that polynome is defined over [0,1].

    """

    N = len(model.cost.signal)
    xobs = np.arange(N) / (N - 1)

    xpred = np.linspace(0, 1, 500)
    ypred = polynomial(xpred)

    f, ax = plt.subplots()
    ax.scatter(xobs, model.cost.signal, label="Observations")
    ax.plot(xpred, ypred, label="Adjusted Polynomial", color="orange")
    if model.bkps is not None:
        idx = np.array(model.bkps, dtype=int) - 1
        ax.scatter(
            model.bkps / (N - 1),
            model.cost.signal[idx],
            c="red",
            label="True change-points",
        )

    if show_predicted:
        ymin = min(model.cost.signal)
        ymax = max(model.cost.signal)
        ax.vlines(
            x=polynomial.x,
            ymin=ymin,
            ymax=ymax,
            ls="--",
            lw=1,
            label="Predicted change-points",
        )
    if show_states is not None:
        # pstates = np.unique(poly(bkps/(N-1)))
        xmin = 0
        xmax = 1
        ax.hlines(
            y=show_states,
            xmin=xmin,
            xmax=xmax,
            ls="--",
            lw=0.5,
        )

    if legend:
        plt.legend()
    plt.show()


def get_polynomial_from_penalized_model(model,y, method='scipy', s=None) -> interpolate.PPoly:
    """
    Reconstruct the approximating polynomial.
    """
    if method=='scipy':
        x = np.linspace(0,1,model.n_points,endpoint=False)
        values = model.states[model.state_idx_sequence]
        bkps = model.bkps
        if len(bkps) > 0:
            if bkps[0] < 3:
                bkps[0] = 3
            if bkps[-1] > model.n_points-3:
                bkps[-1] = model.n_points-3
        if s == None:
            s = 0.1 # default value
        t = np.hstack(tup=(np.array([0,0,0]), bkps/model.n_points))
        t = np.hstack((t, np.array([1,1,1])))
        tck = interpolate.make_splrep(x,y,xb=x.min(),xe=x.max(),k=2,t=t, s=s)
        polynomial = interpolate.PPoly.from_spline(tck)

    else:
        intbkps = model.bkps.astype(int)
        n_points = model.n_points
        step_size = 1 / n_points
        n_steps = np.diff(intbkps)  # between change points
        L = len(intbkps) - 1  # Nb of polynomial segments to parametrize

        # First segment parameters
        segment_start_speed = model.speed_path_mat[0, model.state_idx_sequence[0]]
        speed = np.array([segment_start_speed])
        acc = np.array(
            []
        )  # Acceleration here depends on final speed, it is added afterwards

        # Rest of the points
        # Iterate over polynomial segments
        for bkp_idx in range(1, L):
            # Get the end speed for this segment
            segment_start_speed = model.speed_path_mat[
                intbkps[bkp_idx], model.state_idx_sequence[bkp_idx]
            ]
            speed = np.append(
                arr=speed,
                values=segment_start_speed,
            )
            # Compute acceleration for the previous segment
            prev_seg_acc = np.array(
                [
                    (speed[bkp_idx] - speed[bkp_idx - 1])
                    / (2 * n_steps[bkp_idx - 1] * step_size)
                ]
            )
            acc = np.concatenate((acc, prev_seg_acc))

        # Final speed and acceleration
        final_speed = model.speed_path_mat[intbkps[L], model.state_idx_sequence[L]]
        prev_seg_acc = np.array(
            [(final_speed - speed[L - 1]) / (2 * n_steps[L - 1] * step_size)]
        )
        acc = np.concatenate((acc, prev_seg_acc))

        # Polynomial parameters and construction
        c = np.array([acc, speed, model.states[model.state_idx_sequence[:-1]]])
        x = intbkps
        polynomial = interpolate.PPoly(c=c, x=x * step_size)

    return polynomial


def get_polynomial_from_constrained_model(model, y, method='scipy') -> interpolate.PPoly:
    """
    Reconstruct the approximating polynomial.
    """
    if method=='scipy':
        x = np.linspace(0,1,model.n_points,endpoint=False)
        values = model.states[model.state_idx_sequence]
        bkps = model.bkps
        if len(bkps) > 0:
            if bkps[0] < 3:
                bkps[0] = 3
            if bkps[-1] > model.n_points-3:
                bkps[-1] = model.n_points-3
        t = np.hstack((np.array([0,0,0]), bkps/model.n_points))
        t = np.hstack((t, np.array([1,1,1])))
        tck = interpolate.make_lsq_spline(x,y,t=t,k=2,)
        polynomial = interpolate.PPoly.from_spline(tck)

    else:
        intbkps = model.knots.astype(int)
        n_points = model.n_points
        step_size = 1 / n_points
        n_steps = np.diff(intbkps)  # between change points
        L = len(intbkps) - 1  # Nb of polynomial segments to parametrize

        # First segment parameters
        segment_start_speed = model.speed_path_mat[0, 0, model.state_idx_sequence[0]]
        speed = np.array([segment_start_speed])
        acc = np.array(
            []
        )  # Acceleration here depends on final speed, it is added afterwards

        # Rest of the points
        # Iterate over polynomial segments
        for bkp_idx in range(1, L):
            # Get the end speed for this segment
            segment_start_speed = model.speed_path_mat[
                bkp_idx, intbkps[bkp_idx-1], model.state_idx_sequence[bkp_idx]
            ]
            speed = np.append(
                arr=speed,
                values=segment_start_speed,
            )
            # Compute acceleration for the previous segment
            prev_seg_acc = np.array(
                [
                    (speed[bkp_idx] - speed[bkp_idx - 1])
                    / (2 * n_steps[bkp_idx - 1] * step_size)
                ]
            )
            acc = np.concatenate((acc, prev_seg_acc))

        # Final speed and acceleration
        final_speed = model.speed_path_mat[L, intbkps[L], model.state_idx_sequence[L]]
        prev_seg_acc = np.array(
            [(final_speed - speed[L - 1]) / (2 * n_steps[L - 1] * step_size)]
        )
        acc = np.concatenate((acc, prev_seg_acc))

        # Polynomial parameters and construction
        c = np.array([acc, speed, model.states[model.state_idx_sequence[:-1]]])
        x = intbkps
        polynomial = interpolate.PPoly(c=c, x=x * step_size)

    return polynomial


def draw_bkps(n_bkps, n_samples, normalized, random_state):
    """
    Draw random changepoint indexes.

    Args:
    n_bkps (int) : Number of break points in the sample.
    n_samples (int) : Total points in the signal. It will add last point as final break point.
    random_state (int) : Random seed for sampling.

    Returns:
    bkps (np.array) : A (n_bkps + 1)-long array with the breakpoints, where bkps_int[-1] = n_samples.
    """
    rng = np.random.default_rng(random_state)
    alpha = np.ones(n_bkps + 1, dtype=int) * n_samples
    bkps_float = rng.dirichlet(alpha)
    if normalized:
        bkps = np.cumsum(bkps_float).astype(float).tolist()
        bkps[-1] = 1
    else:
        bkps = np.cumsum(bkps_float * n_samples).astype(int).tolist()
        bkps[-1] = n_samples
    return bkps


def generate_pw_quadratic(
    n_bkps: int = 5,
    n_points: int = 1000,
    normalized=True,
    random_state: int = None,
    delta: float = None,
    strategy: str = None,
) -> interpolate.PPoly:
    """
    Randomly generates a piecewise quadratic polynomial.

    Args:
        n_bkps (int): Number of change-points.
        n_points (int): Total points in the signal.
        normalized (bool) : If True, change-points are converted to [0,1].
        random_state (int) : Random seed for sampling the changes.
        delta (float): Minimum jump-size.
        strategy (string): 'exact' jumps of size 'delta', exactly. Others, not implemented yet.

    Returns:
        poly (scipy.interpolate.Polynomial) : Randomly generated polynomial over the interval [0,1] if normalized,
            or [0, n_points-1] if not normalized.

    A note on how to use in multiple tests
    ---------------------------------------
    If you're going to perform multiple tests with different n_points, you should
    first generate the polynomials with the coarsest one of all. In that manner, you ensure
    that the breakpoints that are sampled belong to all of the discretisations.
    For example, if I want to do experiences with 500, 1000 and 2000 points in [0,1]
    the polynomials should be generated with n_points = 500. This way, the change-points will
    belong to [0, 1/500, 2/500, ..., 499/500] all of which are also in the discretisation
    of [0,1] with 1k and 2k points. Adding more points to the same interval grabs the "middle points"
    and so we would end up with changes in places that are not part of our discretisation.
    """
    n_poly = n_bkps + 1

    if random_state is not None:
        np.random.seed(seed=random_state)
    else:
        np.random.seed(12)

    # Randomly sample breakpoints from the integer range
    x_breaks = draw_bkps(
        n_bkps=n_bkps,
        n_samples=n_points,
        normalized=normalized,
        random_state=random_state,
    )
    x_breaks = np.hstack((0, x_breaks))

    # Need to ensure that the chosen bkps are part of the discretisation.
    # Generate the discretization and get the closest point to the random breaks
    if normalized:
        coarse_x = np.linspace(start=0, stop=1, num=n_points, endpoint=False)
    else:
        coarse_x = np.linspace(start=0, stop=n_points, num=n_points, endpoint=False)
    break_idxs = np.argmin(
        np.abs(coarse_x.reshape((n_points, 1)) - x_breaks[1:-1].reshape((1, n_bkps))),
        axis=0,
    )
    x_breaks[1:-1] = coarse_x[break_idxs]

    # Start coefficient building
    coefficients = np.zeros((3, n_poly), dtype=float)

    # Initial conditions for the first piece
    coefficients[0, 0] = np.random.randint(-5, 5)
    coefficients[1, 0] = np.random.uniform(-5, 5)
    coefficients[2, 0] = np.random.uniform(-5, 5)
    poly = interpolate.PPoly(c=coefficients, x=x_breaks)
    
    a_i = coefficients[0, 0]
    for i in range(1, n_poly):
        # Previous segment endpoint    
        x_curr = x_breaks[i]

        # Set the constant term for continuity
        c_i = poly(x_curr - 1e-7)

        # Set the linear term for smoothness (first derivative continuity)
        b_i = poly.derivative()(x_curr - 1e-7)

        # Set random quadratic term and avoid repetition
        # to ensure the nb of changes
        # sample with minimum jump size
        if delta:
            if strategy == "equal":
                jump_i = (
                    delta * np.random.choice(a=[-1, 1], size=[1], replace=True)[0]
                )
                
            elif strategy == "geq":
                jump_i = np.random.uniform(low=-3 * delta, high=3 * delta)
                while np.abs(jump_i) < delta:
                    jump_i = np.random.uniform(low=-3 * delta, high=3 * delta)
            else:
                raise (Exception("Not implemented error"))
            a_i = a_i + jump_i
        else:
            a_i = np.random.randint(low=-5, high=5)
            # While loop to avoid repetition
            while a_i == coefficients[0, i - 1]:
                a_i = np.random.randint(low=-5, high=5)

        coefficients[:, i] = [a_i, b_i, c_i]
    final_poly = interpolate.PPoly(c=coefficients, x=x_breaks)
    return final_poly

def get_polynomial_knots_and_states(model):
     
    knots = model.knots # includes 0 and n_points
    state_sequence = model.states[model.state_idx_sequence] # includes first and last
    n_points = model.n_points
    speeds = model.initial_speeds # for the moment it's only 1 speed, may need to refactor later

    coeff = np.empty((3,len(knots)-1)) # we don't need coeffs for the segment after the last knot
    coeff[2,:] = state_sequence[:-1]
    coeff[1,0] = speeds[0]

    dknots = np.diff(knots/n_points)
    p_i = state_sequence[0]
    dp = np.diff(state_sequence)
    v_i = speeds[0]

    for i in range(len(dknots)-1):        
        a_i =  (dp[i])/(dknots[i])**2 - v_i / dknots[i]
        coeff[0,i] = a_i
        # Exits
        v_i = v_i + 2 * a_i * dknots[i]
        coeff[1,i+1] = v_i
        p_i = state_sequence[i+1]
    coeff[0,len(dknots)-1] = (dp[len(dknots)-1])/(dknots[len(dknots)-1])**2 - v_i / dknots[len(dknots)-1]
    # Coeff[1/2, last] has the final point and exit speed, but
    # we cannot get an acceleration there (nor are we interested in it)
    # but it is eassier to loop around this way
    coeff = coeff[:,:]
    knots = knots[:]/n_points
    poly = interpolate.PPoly(coeff, knots)
    return poly

def compute_speeds_from_observations(y,pcts=[0.5, 1, 1.5, 2, 2.5]):
    """
    y (np.array) 1-dimensional array with the observations. 
    pcts (list/1d-array) : % of the signal points to take into account
    for the linear regression. Pctgs expressed as integers. 
    """
    x=np.linspace(0,1,len(y),False)
    pct_to_ints = np.round(len(y) * np.array(pcts)/100).astype(int)
    speeds = np.array([])
    for i in pct_to_ints:
        lr = LinearRegression()
        lr.fit(X=x[:i].reshape(-1,1),y=y[:i].reshape(-1,1))
        speed = lr.coef_[0]
        speeds = np.concat((speeds,speed))
    return speeds

def compute_speeds_from_multi_obs(y,pcts=[0.5, 1, 1.5, 2, 2.5])-> np.array:
    """
    Wrapper around get_speeds to compute over the matrix of observations directly.

    y (2d-np.array): N samples x T array of observations
    pcts (list/np.array): Percentages expressed as integers
    """
    speeds = np.apply_along_axis(compute_speeds_from_observations,1,y,pcts)
    return speeds

def state_generator(signal,n_states=5, pct=0.05, local=True):
    """
    Generates a grid of states around the observations. 

    Parameters:
    signal (np.array) : Array of observations
    n_states (int) : Number of states to fit.
    pct (float) : Percentage of the value range to be taken to form the states-grid.
    
    Returns:
    states (np.NDarray) : (n_obs+1, n_states)-shaped array with the states for each observation.
    """
    if signal.shape[0]== 1:
        m = len(signal)
        states = np.zeros((m+1, n_states))
        
        # Compute the inverval around the points
        max_signal = np.max(signal)
        min_signal = np.min(signal)
        interval = np.abs(max_signal - min_signal)
        delta = interval * pct
        if local:
            for i in range(m):
                start = signal[i] - delta/2
                end = signal[i] + delta/2
                states[i] = np.linspace(start,end, n_states, True)
            states[-1] = states[-2]
        else:
            for i in range(m):
                states[i] = np.linspace(min_signal,max_signal, n_states, True)
            states[-1] = states[-2]
        return states
    else:
        each_side = (n_states - 1) // 2

        signal_length, signal_dims = signal.shape
        states_shape = (signal_length+1, n_states, signal_dims)
        states = np.zeros(shape=states_shape)
        for i in range(each_side, signal_length - each_side - 1):
            
            states[i] = signal[i-each_side:i+each_side + 1]
        for i in range(0,each_side):
            states[i] = signal[0:n_states]
            states[signal_length - (each_side+1)  + i] = signal[-n_states:]
        states[-1] = signal[-n_states:]
        return states