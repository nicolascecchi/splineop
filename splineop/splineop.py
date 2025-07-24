import numpy as np
from numba import njit, objmode
from numba.experimental import jitclass
from numba import int64, float64, int64, float64
from scipy import interpolate
import matplotlib.pyplot as plt
from splineop.costs import *
from timeit import default_timer as timer
from sklearn.linear_model import LinearRegression


splineop_spec_Pen = [("cost", costPenalized.class_type.instance_type)]


@jitclass(splineop_spec_Pen)
class splineOPPenalized(object):
    """A class that allows to solve the splineOP poblem with penalization.

    Methods:
    __init__
    fit
    predict
    backtrack_solution
    """

    n_points: int64
    n_states: int64
    states: float64[:, :, :]
    initial_speeds: float64[:, :]
    bkps: float64[:]
    knots: float64[:]
    state_idx_sequence: int64[:]
    time_path_mat: int64[:, :]
    soc: float64[:, :]
    state_path_mat: int64[:, :]
    speed_path_mat: float64[:, :, :]
    execution_time: float64
    t_start: float64
    ndims: int64

    def __init__(self, cost_fn):
        """Constructor for the class.

        Arguments:
        cost_fn (splineop.costs.costConstrained)

        """
        self.cost = cost_fn

    def fit(
        self,
        signal: np.ndarray,
        states: np.ndarray,
        initial_speeds: np.ndarray,
        sample_size: int = 0 ,
        normalized: bool = True,
    ):
        """
        Stores the attributes  and computes the sums needed
        for solving each error in O(1).
        
        Arguments:
        signal ((N,D)-numpy.ndarray): The input signal. N is the number of observations, D their dimensionality.
        states ((N+1, M, D)numpy.ndarray): The states of the system. M is the number of states. 
                N+1 because of the extra point at the end when computing continuity. 
        initial_speeds ((L, D)numpy.ndarray): The initial speeds of the system. L is the number of speeds. D their dimensionality.
        normalized (bool): (Deprecated, but need to completely remove) Whether the data is normalized. 
        
        Important notice:
            Dimensions must be respected in order for the code to work. This is
            becomes important when working with 1-dimensional signals, care should
            be taken so that the shape is (n_samples, 1) and _NOT_ (n_samples, ). 
        """
        
        
        self.n_points = signal.shape[0]
        self.ndims = signal.shape[1]
        self.n_states = states.shape[1]
        self.states = states  # np.array([_ for _ in set(states)], dtype=np.float64)
        self.initial_speeds = initial_speeds  # np.array([_ for _ in set(initial_speeds)], dtype=np.float64)
        if sample_size <= 0:
            sample_size = self.n_points
        self.cost.fit(signal, states, initial_speeds, sample_size, normalized)
        
    def predict(self, penalty: float) -> None:
        """
        Computes the cost of solving the SplineOP problem with a given penalty.

        Arguments
        penalty (float): The penalty term. Bigger penalties generate less change points.
        
        In the code *end* index is excluded. Is a "forward looking" polynomial point.
        It kind of refers to "here's the tip of the polynomial". 
        So, if we are talking about a polynomial that begins at index 0,
        having end=10 means "you have seen 10 points, [0,1,..,9] and the tip of the polynomial
        is at position "almost" 10.
        So there are 10 unit-segments, 10 observed points. 
        If now mid=1, we observed [1,...,9], the tip is in position 10.
        We have observed 10-1 = 9 points and have also 9 unit intervals. 
        This is the reason why the states have an "extra time dimension" so that we can
        "see" all the signal, and place the tip of the polynomial at the end of the last unit-segment. 

        The non-inclusion of the point indexed by [end] is managed inside the 
        <Cost> class methods.  
        """
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
            shape=(self.n_points + 1, self.n_states, self.ndims), dtype=np.float64
        )
        with objmode(t_start="float64"):
            t_start = timer()
        for end in range(1, self.n_points + 1):
            # [end] index will take last value [n_points]\
            # which is NOT in [signal], but in [states]
            for p_end_idx in range(self.n_states):
                (
                    self.soc[end, p_end_idx],
                    self.speed_path_mat[end, p_end_idx],
                    self.state_path_mat[end, p_end_idx],
                    self.time_path_mat[end, p_end_idx],
                    opt_start_speed,  # Possibly need to check this.
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
                        opt_start_speed
                    )
        with objmode(t_end="float64"):
            t_end = timer()
        self.execution_time = t_end - t_start
        return self.backtrack_solution()

    def backtrack_solution(self) -> tuple[np.ndarray, np.ndarray]:
        """Finds the state and time sequence that optimizes the splineOP problem."""
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


splineop_spec_Constrained = [("cost", costConstrained.class_type.instance_type)]
@jitclass(splineop_spec_Constrained)
class splineOPConstrained(object):
    """A class that allows to solve the splineOP problem with a fixed number of breaks.

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
    initial_speeds: float64[:, :]
    bkps: int64[:]
    knots: int64[:]
    state_idx_sequence: int64[:]
    time_path_mat: int64[:, :, :]
    soc: float64[:, :, :]
    state_path_mat: int64[:, :, :]
    speed_path_mat: float64[:, :, :, :]
    execution_time: float64
    execution_time_k: float64[:]
    ndims: int64

    def __init__(self, cost_fn):
        """Constructor for the class.

        Arguments:
        cost_fn (splineop.costs.costConstrained)

        """
        self.cost = cost_fn

    def fit(
        self,
        signal: np.ndarray,
        states: np.ndarray,
        initial_speeds: np.ndarray,
        sample_size: int = 0,
        normalized: bool = True,
    ):
        """
        Stores the attributes  and computes the sums needed
        for solving each error in O(1).

        Arguments:
        signal ((N,D)-numpy.ndarray): The input signal. N is the number of observations, D their dimensionality.
        states ((N+1, M, D)numpy.ndarray): The states of the system. M is the number of states. 
                N+1 because of the extra point at the end when computing continuity. 
        initial_speeds ((L, D)numpy.ndarray): The initial speeds of the system. L is the number of speeds. D their dimensionality.
        normalized (bool): (Deprecated, but need to completely remove) Whether the data is normalized. 
        
        Important notice:
            Dimensions must be respected in order for the code to work. This is
            becomes important when working with 1-dimensional signals, care should
            be taken so that the shape is (n_samples, 1) and _NOT_ (n_samples, ).
        """
        
        self.n_points = signal.shape[0]
        self.ndims = signal.shape[1]
        self.n_states = states.shape[1]
        self.states = states  # np.array([_ for _ in set(states)], dtype=np.float64)
        self.initial_speeds = initial_speeds  # np.array([_ for _ in set(initial_speeds)], dtype=np.float64)
        if sample_size<=0:
            sample_size = self.n_points
        self.cost.fit(signal, states, initial_speeds, sample_size, normalized)        

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
        self.soc[0] = float(0.0)  # Dummy
        self.soc[1] = np.inf  # Puts infinite weight to segments w/o change
        # to avoid having sthing like [0..2][3...T]
        # Avoiding a very short first segment for sure.
        self.time_path_mat = np.empty(
            shape=(K + 2, self.n_points + 1, self.n_states), dtype=np.int64
        )
        self.state_path_mat = np.empty(
            shape=(K + 2, self.n_points + 1, self.n_states), dtype=np.int64
        )
        self.speed_path_mat = np.empty(
            shape=(K + 2, self.n_points + 1, self.n_states, self.ndims),
            dtype=np.float64,
        )
        self.execution_time_k = np.zeros(shape=(K + 2), dtype=np.float64)

        for k in range(1, K + 2):  # nb of segments
            with objmode(t_start_k="float64"):
                t_start_k = timer()
            for end in range(k, self.n_points + 1):  # nb of points seen
                for p_end_idx in range(self.n_states):  # each state
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
            with objmode(t_end_k="float64"):
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
            [int(np.argmin(self.soc[K - 1, -1]))], dtype=np.int64
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
        assert K <= self.soc.shape[0] - 2
        K = K + 2  # I do +2 here, and -1 in the def of state_idx_seq below
        # so that the code is resembles more to the "normal" backtrack

        # Get the last time and last position
        t = self.soc.shape[1] - 1  # last item's index
        bkps = np.array([t], dtype=np.int64)
        state_idx_sequence = np.array(
            [int(np.argmin(self.soc[K - 1, -1]))], dtype=np.int64
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
                jump_i = delta * np.random.choice(a=[-1, 1], size=[1], replace=True)[0]

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



def compute_speeds_from_observations(y
                                     , pcts=None
                                     , end_indexes=[3,5,8,13,21]):
    """
    Computes set of initial speeds from the observations. 

    Params:
    y (np.array) 1-dimensional array with the observations.
    pcts (list/1d-array) : % of the signal points to take into account
        for the linear regression. Pctgs expressed as integers.
    end_indexes(list/1d-array): List of indexes to use for the linear regression.
    
    Returns:
    speeds (np.array): Array with set of initial speeds. 

    Computes the set of initial speeds as the slope of a linear regression fitted
    over the first [pcts] percentages (or [idx] end_indexes) of points observed.     
    """
    ndims = y.shape[1]
    x = np.linspace(0, 1, len(y), False)

    if pcts is not None:
        nspeeds = len(pcts)
        speeds = np.empty((nspeeds,ndims))
        pct_to_ints = np.ceil(len(y) * np.array(pcts) / 100).astype(int)
        for idx, i in enumerate(pct_to_ints):
            lr = LinearRegression()
            lr.fit(X=x[:i].reshape(-1, 1), y=y[:i])
            speed = lr.coef_.T
            speeds[idx] = speed
    else:
        nspeeds = len(end_indexes)
        speeds = np.empty((nspeeds,ndims))
        for idx, i in enumerate(end_indexes):
            lr = LinearRegression()
            lr.fit(X=x[:i].reshape(-1, 1), y=y[:i])
            speed = lr.coef_.T
            speeds[idx] = speed
    return speeds



