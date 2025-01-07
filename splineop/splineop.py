import numpy as np
from numba import njit
from numba.experimental import jitclass
from numba import int64, float64, int64, float64
from scipy import interpolate
from scipy.stats import dirichlet
import matplotlib.pyplot as plt
from .costs import *

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
        self.n_states = states.shape[0]
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
        self.soc[0, :] = 0
        self.time_path_mat = np.empty(
            shape=(self.n_points + 1, self.n_states), dtype=np.int64
        )
        self.state_path_mat = np.empty(
            shape=(self.n_points + 1, self.n_states), dtype=np.int64
        )
        self.speed_path_mat = np.empty(
            shape=(self.n_points + 1, self.n_states), dtype=np.float64
        )
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
        return self.backtrack_solution()


splineop_spec_Constrained = [("cost", costConstrained.class_type.instance_type)]
@jitclass(splineop_spec_Constrained)
class splineOPConstrained(object):
    n_points: int64
    n_states: int64
    states: float64[:]
    initial_speeds: float64[:]
    bkps: int64[:]
    knots: int64[:]
    state_idx_sequence: int64[:]
    time_path_mat: int64[:, :, :]
    soc: float64[:, :, :]
    state_path_mat: int64[:, :, :]
    speed_path_mat: float64[:, :, :]

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
        self.n_states = states.shape[0]
        self.states = states  # np.array([_ for _ in set(states)], dtype=np.float64)
        self.initial_speeds = initial_speeds  # np.array([_ for _ in set(initial_speeds)], dtype=np.float64)
        self.cost.fit(signal, states, initial_speeds, normalized)

    def predict(self, K):
        """
        Compute the optimal partition for K change-points.

        args
        K (int) Number of changepoints desired by the user, implying K+1 segments.
        """
        # Internally we think the procedure in terms of nb of segments
        # therefore we need dimension K+2 because we index 0 as a dummy
        # and then have indexes 1 through K+1, representing the segments

        # The fist dimension of SOC is used as a dummy with all 0s
        # The first dimensions of the others is a dummy never used
        # but helps in terms of clarity with the indexing.
        self.soc = np.empty(
            shape=(K + 2, self.n_points + 1, self.n_states), dtype=np.float64
                    )
        self.soc[0] = 0  # Dummy
        self.time_path_mat = np.empty(
            shape=(K + 2, self.n_points + 1, self.n_states), dtype=np.int64
        )
        self.state_path_mat = np.empty(
            shape=(K + 2, self.n_points + 1, self.n_states), dtype=np.int64
        )
        self.speed_path_mat = np.empty(
            shape=(K + 2, self.n_points + 1, self.n_states), dtype=np.float64
        )
        for k in range(1, K + 2):
            for end in range(k, self.n_points + 1):
                for p_end_idx in range(self.n_states):
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
        self.backtrack_solution()

    def backtrack_solution(self) -> tuple[np.ndarray, np.ndarray]:
        K = self.soc.shape[0]
        t = self.soc.shape[1] - 1
        bkps = np.array([t], dtype=np.int64)
        state_idx_sequence = np.array(
            [int(np.argmin(self.soc[-1, -1]))], dtype=np.int64
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
        assert K <= self.soc.shape[0]
        K = K+2 # Need to explain more the relation with the dimensions of the matrix
        t = self.soc.shape[1] - 1
        bkps = np.array([t], dtype=np.int64)
        state_idx_sequence = np.array(
            [int(np.argmin(self.soc[K, -1]))], dtype=np.int64
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


def get_polynomial_from_penalized_model(model,y, method='scipy') -> interpolate.PPoly:
    """
    Reconstruct the approximating polynomial.
    """
    if method=='scipy':
        x = np.linspace(0,1,1000,endpoint=False)
        knots = model.knots/model.n_points
        values = model.states[model.state_idx_sequence]
        tck = interpolate.splrep(x,y,xb=0,xe=1,k=2,t=model.bkps/model.n_points, task=-1)
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
        x = np.linspace(0,1,1000,endpoint=False)
        values = model.states[model.state_idx_sequence]
        bkps = model.bkps
        if len(bkps) > 0:
            if bkps[0] < 3:
                bkps[0] = 3
            if bkps[-1] > 997:
                bkps[-1] = 997
            
        tck = interpolate.splrep(x,y,xb=0,xe=1,k=2,t=bkps/model.n_points, task=-1)
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
            print('jump', jump_i)
            print('curr a_i', a_i)
            a_i = a_i + jump_i
            print('new a_i', a_i)
        else:
            a_i = np.random.randint(low=-5, high=5)
            # While loop to avoid repetition
            while a_i == coefficients[0, i - 1]:
                a_i = np.random.randint(low=-5, high=5)

        coefficients[:, i] = [a_i, b_i, c_i]
    final_poly = interpolate.PPoly(c=coefficients, x=x_breaks)
    return final_poly
