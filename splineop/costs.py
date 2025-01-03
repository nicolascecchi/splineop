import numpy as np
from numba import njit
from numba.experimental import jitclass
from numba import int64, float64, int64, float64
from scipy import interpolate
from scipy.stats import dirichlet
import matplotlib.pyplot as plt


@jitclass
class costPenalized(object):
    """
    Class that stores values to compute efficiently the cost of a given segment.
    """

    signal: float64[:]
    states: float64[:]
    n_states: int
    initial_speeds: float64[:]
    normalized: bool
    n_samples: int
    cumsum_y: float64[:]
    cumsum_y_sq: float64[:]
    cumsum_n_y: float64[:]
    cumsum_n_sq_y: float64[:]

    def __init__(self):
        pass

    def compute_crossed_terms(self, s: int, t: int) -> float:
        """
        Computes the 'crossed terms' in the cost of a segment, of the form \sum (n * y),\sum (n^2 * y)

        'effective' refers to the fact that when starting index is not 0, you have to take the difference
        """
        end = t - 1
        if s == 0:
            effective_int_y = self.cumsum_n_y[end]
            effective_int_sq__y = self.cumsum_n_sq_y[end]
            effective_cumsum_y = self.cumsum_y[end]
        else:
            start = s - 1
            effective_int_y = self.cumsum_n_y[end] - self.cumsum_n_y[start]
            effective_int_sq__y = self.cumsum_n_sq_y[end] - self.cumsum_n_sq_y[start]
            effective_cumsum_y = self.cumsum_y[end] - self.cumsum_y[start]

        if self.normalized:
            m = 1 / (self.n_samples)
        else:
            m = 1

        sum_int_y = (effective_int_y - s * effective_cumsum_y) * m
        sum_int_sq_y = (
            effective_int_sq__y * m**2
            - 2 * s * m * sum_int_y
            - (s * m) ** 2 * effective_cumsum_y
        )

        return sum_int_y, sum_int_sq_y

    def fit(
        self,
        signal: np.ndarray,
        states: np.ndarray,
        initial_speeds: np.ndarray,
        normalized: bool,
    ):
        """
        Precomputes and caches in object attributes values to compute the cost.
        """
        self.signal = signal
        self.states = states
        self.n_states = states.shape[0]
        self.initial_speeds = initial_speeds
        self.n_samples = self.signal.shape[0]
        self.normalized = normalized
        self.cumsum_y = np.cumsum(self.signal)
        self.cumsum_y_sq = np.cumsum(self.signal**2)

        # Crossed terms
        T = self.signal.shape[0]
        integers = np.arange(0, T, 1)

        int_times_signal = integers * self.signal
        self.cumsum_n_y = np.cumsum(int_times_signal)

        int_sq_times_signal = integers**2 * self.signal
        self.cumsum_n_sq_y = np.cumsum(int_sq_times_signal)

    def Faulhaber(self, deg: int, n: int) -> int:
        """
        Compute the sum of the first n integers to the power of deg.
        """
        match deg:
            case 0:
                return n
            case 1:
                return n * (n + 1) / 2
            case 2:
                return 1 / 6 * (2 * n**3 + 3 * n**2 + n)
            case 3:
                return 1 / 4 * (n**4 + 2 * n**3 + n**2)
            case 4:
                return 1 / 30 * (6 * n**5 + 15 * n**4 + 10 * n**3 - n)

    def error(
        self,
        start: int,
        end: int,
        p_start_val: float,
        p_end_val: float,
        v_start_val: float,
    ) -> tuple[float, float]:
        """
        Computes the error on the interval [start, end) and the final speed of the associated polynomial.

        args
        start (int): Starting position for the approximating the polynomial P.
        end (int): End position for the approximating polynomial P.
        p_start_val (int): Value of the starting position, such that P(start) = states[p_start_idx].
        p_end_val (int): Value of the ending position, such that P(end) = states[p_end_idx].
        vstart (float): Starting speed, such that P'(start)=vstart.

        returns
        cost_val (float): Value error on the interval.
        vend (float): Speed at the end-point of the interval.
        """
        samples_in_range = end - start
        if self.normalized:
            x_interval = samples_in_range * 1 / (self.n_samples)
        else:
            x_interval = samples_in_range

        a = (p_end_val - p_start_val) / x_interval**2 - v_start_val / x_interval
        b = v_start_val
        c = p_start_val
        # cumsum including left point, excluding right point
        # cumsum[n:k) = cumsum[0:k-1] - cumsum[0:n-1]
        cumsum_high_idx = end - 1  # Not counting rhs point
        cumsum_low_idx = start - 1  # Go one back to really include 'start'
        if start == 0:
            effective_cumsum_sq = self.cumsum_y_sq[cumsum_high_idx]
            effective_cumsum = self.cumsum_y[cumsum_high_idx]
        else:
            effective_cumsum_sq = (
                self.cumsum_y_sq[cumsum_high_idx] - self.cumsum_y_sq[cumsum_low_idx]
            )
            effective_cumsum = (
                self.cumsum_y[cumsum_high_idx] - self.cumsum_y[cumsum_low_idx]
            )

        sum_int_y, sum_int_sq_y = self.compute_crossed_terms(start, end)
        if self.normalized:
            FaulhaberNormalizer = 1 / (self.n_samples)
        else:
            FaulhaberNormalizer = 1
        cost_val = (
            a**2
            * self.Faulhaber(deg=4, n=samples_in_range - 1)
            * FaulhaberNormalizer**4
            + 2
            * a
            * b
            * self.Faulhaber(deg=3, n=samples_in_range - 1)
            * FaulhaberNormalizer**3
            + (2 * a * c + b**2)
            * self.Faulhaber(deg=2, n=samples_in_range - 1)
            * FaulhaberNormalizer**2
            + 2
            * b
            * c
            * self.Faulhaber(deg=1, n=samples_in_range - 1)
            * FaulhaberNormalizer
            - 2 * a * sum_int_sq_y  # np.sum(integers**2 * y)  # self.sum_n_sq_y
            - 2 * b * sum_int_y  # np.sum(integers * y)  # self.sum_n_y
            + effective_cumsum_sq
            - 2 * c * effective_cumsum
            + samples_in_range * c**2
        )
        vend = 2 * (p_end_val - p_start_val) / x_interval - v_start_val
        return cost_val, vend

    def compute_optimal_cost(
        self,
        end: int,
        p_end_idx: int,
        speed_matrix: np.ndarray,
        initial_speeds: np.ndarray,
        soc: np.ndarray,
        penalty: float,
    ) -> tuple[float, float, float, float]:
        """
        Computes the optimal cost of having seen <end> points and ending at position states[p_end_idx].

        Exhaustively computes the cost by evaluating all possible previous change points and initial states.

        args
        end (int): End position for the approximating polynomial P.
        p_end_idx (int):
        speed_matrix (np.ndarray):

        Returns
        optimal cost
        optimal previous speed
        """
        # No break
        curr_optimal_cost_val = np.inf
        curr_optimal_speed_val = np.inf
        curr_optimal_start_state = None
        curr_optimal_time = 0
        curr_optimal_initial_speed = np.inf
        for p_start_idx in range(self.n_states):
            for v_start_val in initial_speeds:
                new_seg_error, new_end_speed = self.error(
                    start=0,
                    end=end,
                    p_start_val=self.states[p_start_idx],
                    p_end_val=self.states[p_end_idx],
                    v_start_val=v_start_val,
                )
                if (
                    new_seg_error < curr_optimal_cost_val
                ):  # What happens if it is equal ??
                    curr_optimal_cost_val = new_seg_error
                    curr_optimal_speed_val = new_end_speed
                    curr_optimal_start_state = p_start_idx
                    curr_optimal_initial_speed = v_start_val
        # Cases with change point
        for p_start_idx in range(self.n_states):
            for mid in range(1, end):
                new_seg_error, new_end_speed = self.error(
                    start=mid,
                    end=end,
                    p_start_val=self.states[p_start_idx],
                    p_end_val=self.states[p_end_idx],
                    v_start_val=speed_matrix[mid, p_start_idx],
                )
                if (
                    soc[mid, p_start_idx] + new_seg_error + penalty
                ) < curr_optimal_cost_val:  # What happens if it is equal ??
                    curr_optimal_cost_val = (
                        soc[mid, p_start_idx] + new_seg_error + penalty
                    )
                    curr_optimal_speed_val = new_end_speed
                    curr_optimal_start_state = p_start_idx
                    curr_optimal_time = mid
        return (
            curr_optimal_cost_val,
            curr_optimal_speed_val,
            curr_optimal_start_state,
            curr_optimal_time,
            curr_optimal_initial_speed,
        )


# splineop_spec = [


# @jitclass
class costConstrained(object):
    """
    Class that stores values to compute efficiently the cost of a given segment.
    """

    signal: float64[:]
    states: float64[:]
    n_states: int
    initial_speeds: float64[:]
    normalized: bool
    n_samples: int
    cumsum_y: float64[:]
    cumsum_y_sq: float64[:]
    cumsum_n_y: float64[:]
    cumsum_n_sq_y: float64[:]

    def __init__(self):
        pass

    def compute_crossed_terms(self, s: int, t: int) -> float:
        """
        Computes the 'crossed terms' in the cost of a segment, of the form \sum (n * y),\sum (n^2 * y)

        'effective' refers to the fact that when starting index is not 0, you have to take the difference
        """
        end = t - 1
        if s == 0:
            effective_int_y = self.cumsum_n_y[end]
            effective_int_sq__y = self.cumsum_n_sq_y[end]
            effective_cumsum_y = self.cumsum_y[end]
        else:
            start = s - 1
            effective_int_y = self.cumsum_n_y[end] - self.cumsum_n_y[start]
            effective_int_sq__y = self.cumsum_n_sq_y[end] - self.cumsum_n_sq_y[start]
            effective_cumsum_y = self.cumsum_y[end] - self.cumsum_y[start]

        if self.normalized:
            m = 1 / (self.n_samples)
        else:
            m = 1

        sum_int_y = (effective_int_y - s * effective_cumsum_y) * m
        sum_int_sq_y = (
            effective_int_sq__y * m**2
            - 2 * s * m * sum_int_y
            - (s * m) ** 2 * effective_cumsum_y
        )

        return sum_int_y, sum_int_sq_y

    def fit(
        self,
        signal: np.ndarray,
        states: np.ndarray,
        initial_speeds: np.ndarray,
        normalized: bool,
    ):
        """
        Precomputes and caches in object attributes values to compute the cost.
        """
        self.signal = signal
        self.states = states
        self.n_states = states.shape[0]
        self.initial_speeds = initial_speeds
        self.n_samples = self.signal.shape[0]
        self.normalized = normalized
        self.cumsum_y = np.cumsum(self.signal)
        self.cumsum_y_sq = np.cumsum(self.signal**2)

        # Crossed terms
        T = self.signal.shape[0]
        integers = np.arange(0, T, 1)

        int_times_signal = integers * self.signal
        self.cumsum_n_y = np.cumsum(int_times_signal)

        int_sq_times_signal = integers**2 * self.signal
        self.cumsum_n_sq_y = np.cumsum(int_sq_times_signal)

    def Faulhaber(self, deg: int, n: int) -> int:
        """
        Compute the sum of the first n integers to the power of deg.
        """
        match deg:
            case 0:
                return n
            case 1:
                return n * (n + 1) / 2
            case 2:
                return 1 / 6 * (2 * n**3 + 3 * n**2 + n)
            case 3:
                return 1 / 4 * (n**4 + 2 * n**3 + n**2)
            case 4:
                return 1 / 30 * (6 * n**5 + 15 * n**4 + 10 * n**3 - n)

    def error(
        self,
        start: int,
        end: int,
        p_start_val: float,
        p_end_val: float,
        v_start_val: float,
    ) -> tuple[float, float]:
        """
        Computes the error on the interval [start, end) and the final speed of the associated polynomial.

        args
        start (int): Starting position for the approximating the polynomial P.
        end (int): End position for the approximating polynomial P.
        p_start_val (int): Value of the starting position, such that P(start) = states[p_start_idx].
        p_end_val (int): Value of the ending position, such that P(end) = states[p_end_idx].
        vstart (float): Starting speed, such that P'(start)=vstart.

        returns
        cost_val (float): Value error on the interval.
        vend (float): Speed at the end-point of the interval.
        """
        samples_in_range = end - start
        if self.normalized:
            x_interval = samples_in_range * 1 / (self.n_samples)
        else:
            x_interval = samples_in_range

        a = (p_end_val - p_start_val) / x_interval**2 - v_start_val / x_interval
        b = v_start_val
        c = p_start_val
        # cumsum including left point, excluding right point
        # cumsum[n:k) = cumsum[0:k-1] - cumsum[0:n-1]
        cumsum_high_idx = end - 1  # Not counting rhs point
        cumsum_low_idx = start - 1  # Go one back to really include 'start'
        if start == 0:
            effective_cumsum_sq = self.cumsum_y_sq[cumsum_high_idx]
            effective_cumsum = self.cumsum_y[cumsum_high_idx]
        else:
            effective_cumsum_sq = (
                self.cumsum_y_sq[cumsum_high_idx] - self.cumsum_y_sq[cumsum_low_idx]
            )
            effective_cumsum = (
                self.cumsum_y[cumsum_high_idx] - self.cumsum_y[cumsum_low_idx]
            )

        sum_int_y, sum_int_sq_y = self.compute_crossed_terms(start, end)
        if self.normalized:
            FaulhaberNormalizer = 1 / (self.n_samples)
        else:
            FaulhaberNormalizer = 1
        cost_val = (
            a**2
            * self.Faulhaber(deg=4, n=samples_in_range - 1)
            * FaulhaberNormalizer**4
            + 2
            * a
            * b
            * self.Faulhaber(deg=3, n=samples_in_range - 1)
            * FaulhaberNormalizer**3
            + (2 * a * c + b**2)
            * self.Faulhaber(deg=2, n=samples_in_range - 1)
            * FaulhaberNormalizer**2
            + 2
            * b
            * c
            * self.Faulhaber(deg=1, n=samples_in_range - 1)
            * FaulhaberNormalizer
            - 2 * a * sum_int_sq_y  # np.sum(integers**2 * y)  # self.sum_n_sq_y
            - 2 * b * sum_int_y  # np.sum(integers * y)  # self.sum_n_y
            + effective_cumsum_sq
            - 2 * c * effective_cumsum
            + samples_in_range * c**2
        )
        vend = 2 * (p_end_val - p_start_val) / x_interval - v_start_val
        return cost_val, vend

    def compute_optimal_cost(
        self,
        end: int,
        p_end_idx: int,
        speed_matrix: np.ndarray,
        initial_speeds: np.ndarray,
        soc: np.ndarray,
        k: int,
    ) -> tuple[float, float, float, float]:
        """
        Computes the optimal cost of having seen <end> points and ending at position states[p_end_idx].

        Exhaustively computes the cost by evaluating all possible previous change points and initial states.

        args
        end (int): End position for the approximating polynomial P.
        p_end_idx (int):
        speed_matrix (np.ndarray):

        Returns
        optimal cost
        optimal previous speed
        """
        # No break
        curr_optimal_cost_val = np.inf
        curr_optimal_speed_val = np.inf
        curr_optimal_start_state = None
        curr_optimal_time = 0

        # K = 1, cad 0 changes
        if k == 1:
            for p_start_idx in range(self.n_states):
                for v_start_val in initial_speeds:
                    new_seg_error, new_end_speed = self.error(
                        start=0,
                        end=end,
                        p_start_val=self.states[p_start_idx],
                        p_end_val=self.states[p_end_idx],
                        v_start_val=v_start_val,
                    )
                    if (
                        new_seg_error < curr_optimal_cost_val
                    ):  # What happens if it is equal ??
                        curr_optimal_cost_val = new_seg_error
                        curr_optimal_speed_val = new_end_speed
                        curr_optimal_start_state = p_start_idx
        # Cases with change point
        else:
            for p_start_idx in range(self.n_states):
                for mid in range(k - 1, end):
                    new_seg_error, new_end_speed = self.error(
                        start=mid,
                        end=end,
                        p_start_val=self.states[p_start_idx],
                        p_end_val=self.states[p_end_idx],
                        v_start_val=speed_matrix[mid, p_start_idx],
                    )
                    if (
                        soc[mid, p_start_idx] + new_seg_error
                    ) < curr_optimal_cost_val:  # What happens if it is equal ??
                        curr_optimal_cost_val = soc[mid, p_start_idx] + new_seg_error
                        curr_optimal_speed_val = new_end_speed
                        curr_optimal_start_state = p_start_idx
                        curr_optimal_time = mid
        return (
            curr_optimal_cost_val,
            curr_optimal_start_state,
            curr_optimal_time,
            curr_optimal_speed_val,
        )
