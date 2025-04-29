import splineop.splineop as sop
import splineop.costs as costs
import numpy as np
from scipy import stats
import pickle
import time
import cvxpy as cp
from scipy import interpolate
from scipy.interpolate import splrep, BSpline
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Arrow, Circle


def moving_average(a, n=2):
    """
    Compute moving average of array 'a' over a window of size 'n'.

    Credit to @Jaime from https://stackoverflow.com/questions/14313510/how-to-calculate-rolling-moving-average-using-python-numpy-scipy
    """

    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1 :] / n


def get_states(poly, signal, heuristic, nstates):
    """
    Generate the discrete states with which the SplineOP algorithm will be computed.

    Args:
        poly (scipy.interpolate.Polynomial): Polynomial from which to get the states.
        xmax (int): Maximum x value.
        nstates (int): Number of points to sample as states.
                       Ignored if heuristic is 'truth'.
        heuristic (str): One of 'uniform', 'qtiles' and 'truth'
                - uniform: Discrete uniform sampling between ymin and ymax
                - qtiles: Median value of the q-quantiles of the sample
                - truth: true values
        normalized (bool) : Wheter the generating polynomial has been normalized.

    Returns:
        states (numpy.array): Array with the values of the states based on the chosen heuristic.

    """

    match heuristic:
        case "uniform":
            ymin = np.min(signal)
            ymax = np.max(signal)
            states = np.linspace(ymin, ymax, nstates)
        case "qtiles":
            quantiles = np.percentile(signal, np.linspace(0, 100, nstates + 1))
            states = moving_average(quantiles, n=2)
        case "truth":
            # poly.x includes the rightmost point of the interval, so that it should
            # be the point after seing the N points of the signal.
            states = poly(poly.x)
    states = np.array(states, dtype=np.float64)
    return states


def get_speeds(poly, signal, heuristic, n=5):
    """
    Generates the initial speeds according to the selected heuristic.

    Args:
        poly (scipy.interpolate.Polynomial): Polynomial from which to extract true derivatives.
        heuristic (str): One of 'spline' or 'truth'.
            linreg: Computes a quadratic spline (ver como) with the first N points and gets the derivative
            truth: Get the derivative from poly
    Returns:
        speeds (numpy.array): Array with the values of the discretized speeds based on the chosen heuristic.
    """
    match heuristic:
        case "linreg":
            speeds = stats.linregress(x=np.arange(0, n), y=signal[0:n])[0]
        case "truth":
            true_changes = poly.x[0]
            speeds = np.unique(poly.derivative()(true_changes))
    # Prevent speeds = (nb) and has no shape
    if speeds.shape == ():
        speeds = np.array([speeds])
    speeds = np.array(speeds, dtype=np.float64)
    return speeds


def noise_signal(signal, sigma, seed):
    """
    Add noise to the signal.

    """
    if sigma == 0:
        return signal
    else:
        noise_shape = signal.shape
        noise = stats.norm(loc=0, scale=sigma).rvs(size=noise_shape, random_state=seed)
        noised_signal = signal + noise
        return noised_signal


def load_and_compute(
    datapath,
    n_bkps,
    n_points,
    seed,
    noise_idx,
    pos_heuristic,
    speed_heuristic,
    nstates=None,
):
    """
    Pipeline for generating the necessary data for running an experience.

    Args:
    n_bkps (int) : Number of breakpoints in the signal.
    seed (int) : Random seed
    noise_idx (int) : Level of noise 0 to 3, increasing noise.
    pos_heuristic (str) : Heuristic for discretising the positions.
    speed_heuristic (str) : Heuristic for discretising the speed.
    nstates (int) : Number of positions to generate with the position heuristic.

    Returns:


    """
    poly, noised_signal, clean_signal = load_PolyObs(
        datapath=datapath,
        n_bkps=n_bkps,
        n_points=n_points,
        seed=seed,
        noise_idx=noise_idx,
    )
    if nstates is not None:
        states = get_states(
            poly=poly, signal=noised_signal, heuristic=pos_heuristic, nstates=nstates
        )
        speeds = get_speeds(poly=poly, signal=noised_signal, heuristic=speed_heuristic)
        return poly, noised_signal, clean_signal, states, speeds
    else:
        return poly, noised_signal, clean_signal


def predict_pipeline(signal, positions, speeds, pen=None, normalized=True, K=None):
    """
    Fits model to <signal> and predicts change points.

    Args:
        signal (np.array): Signal to work on.
        positions (np.array): Discretization of positions.
        speeds (np.array): Discretization of initial speeds.
        pen (float): Penalty term.
        normalized (bool) : Wether the signal lives in [0,1) or not.

    Returns:
        model (sop.splineOP): SplineOP model fit to the signal with given data.

    """
    if K:
        cost = costs.costConstrained()
        model = sop.splineOPConstrained(cost)

        model.fit(signal, positions, speeds, normalized)
        t0 = time.process_time()
        model.predict(K)
        deltat = time.process_time() - t0
    else:
        cost = costs.costPenalized()
        model = sop.splineOPPenalized(cost)

        model.fit(signal, positions, speeds, normalized)
        t0 = time.process_time()
        model.predict(pen)
        deltat = time.process_time() - t0
    return model, deltat


def load_PolyObs(datapath, n_bkps, n_points, seed, noise_idx):
    """
    Load polynomial and noised signal.

    A polynomial is uniquely determined by n_bkps and seed.
    A signal is uniquely determined by n_bkps, noise_lvl and seed.

    Args:
    n_bkps (int): Number of changes. Must be in [0, 1, 2, 3, 4, 5].
    seed (int): Seed that generated the polynomial. Seed \in range(50).
    noise_lvl (int): From 0 to 3, increasing the level of noise.

    Returns:
    poly (scipy.interpolate.Polynomial): Polynomial object generating the signal.
    signal (np.array): Array with the observations.

    """
    with open(datapath + f"/polynomes/polynome{seed}_bkps{n_bkps}.pkl", "rb") as f:
        poly = pickle.load(f)
    noised_signal = np.loadtxt(
        datapath
        + f"/signals/noise{noise_idx+1}/noised_{n_points}points_{n_bkps}bkps.txt",
        dtype=np.float64,
        delimiter=";",
    )[seed]
    clean_signal = np.loadtxt(
        datapath + f"/signals/clean_signals/signals_{n_points}points_{n_bkps}bkps.txt",
        dtype=np.float64,
        delimiter=";",
    )[seed]
    return poly, noised_signal, clean_signal


def trend_filtering_pwq(signal: np.ndarray, vlambda: float):
    """
    L1 trend-filtering algorithm.

    input
    signal :: np.array
    vlambda :: penalty term
    """
    # quality check
    if len(signal.shape) == 1:
        signal = signal.reshape(len(signal), 1)
    if len(signal.shape) > 2 or 1 not in signal.shape:
        raise Exception(
            f"Signal shape error. Expected (npoints,1), received {signal.shape}"
        )
    if signal.shape[0] < signal.shape[1]:
        signal = signal.T
    # Setup
    degree = 2
    u = cp.Variable(shape=signal.shape, name="u")
    obj = cp.Minimize(
        cp.sum_squares(signal - u)
        + vlambda * cp.mixed_norm(cp.diff(u, k=degree + 1), p=2, q=1)
    )
    prob = cp.Problem(obj)
    # Solve
    t0 = time.process_time()
    prob.solve(solver=cp.CLARABEL, verbose=False)  # modify solver if too long
    run_time = time.process_time() - t0

    if prob.status != cp.OPTIMAL:
        raise Exception("Solver did not converge!")
    signal_smoothed = u.value.flatten()
    return signal_smoothed, run_time

def compute_psnr(signal,prediction):
    """
    Peak signal to noise ratio. 
    PSNR = 20 * log_10 (Maximum Intensity) - 10 * log_10 (MSE) 

    Input
    signal (np.ndarray): Of shape (N samples, N dims).
    prediction (np.ndarray): Of shape (N samples, N dims).

    Output
    psnr (float): The PSNR computed between the prediction and the real values. 

    For the Maximum value of the signal we take it empirically from all the dimensions.
    For the MSE, we compute the average over dimensions and time. 

    """
    maxi = np.max(signal)
    mse = np.mean((signal-prediction)**2)
    psnr = 20 * np.log10(maxi) - 10*np.log10(mse)
    return psnr

def compute_bic(signal, K):
    """
    Computes the BIC of the signal with a given **number of segments** K.

    input
    signal (np.ndarray): Of shape (N samples, N dims). 
    K (int): Number of segments being fit.  

    returns 
    bic (float) : Penalization to fit the model. 
    """
    sigma = np.mean(np.std(signal, axis=0))
    n_params = 1 + 2* K
    T = len(signal)
    bic = sigma**2 * n_params * np.log(T)
    return bic


def spline_approximation(data, n_points, pen, degree=2):
    time_array = np.linspace(
        0, 1, num=n_points
    )  # set the interval over which to fit the spline
    t0 = time.process_time()
    tck, error, _, _ = splrep(
        x=time_array, y=signal, k=degree, task=0, s=pen, full_output=1  #  fit spline
    )
    run_time = time.process_time() - t0
    index_break_points = np.round(np.unique(tck[0]) * n_points)

    return index_break_points, error, run_time


def bkps_to_spline(yobs, bkps, deg=2):
    """
    Given bkps and observations, constructs the BSpline.antiderivative

    Args
        yobs: Observed signal to fit.
        bkps: Knots for the spline, not including 0 and 1.
        deg: Degree of the polynomial.

    Returns
        poly (scipy.interpolate.BSpline): Spline of degree deg fitted to yobs.


    bkps must be in (0,1), not include extremes.

    """
    try:
        x = np.linspace(0, 1, len(yobs), endpoint=False)
        t, c, k = splrep(
            x=x,
            y=yobs,
            xb=0,
            xe=1,
            k=deg,
            t=bkps,
        )
        poly = BSpline(c=c, t=t, k=k)
    except:
        x = np.linspace(0, 1, len(yobs), endpoint=True)
        t, c, k = splrep(
            x=x,
            y=yobs,
            xb=0,
            xe=1,
            k=deg,
            t=bkps,
        )
        poly = BSpline(c=c, t=t, k=k)
    return poly


def see_predictions(n_points, n_bkps, seed, noise_idx, multiplier, tftol, dfpivot):
    """
    Plots observations, true signal, and the 3 predictions we are comparing.


    n_points, n_bkps, seed, noise_idx to identify the signal.
    pen(alty) to run the models.
    """
    noises = {0: 0.1, 1: 0.13, 2: 0.16}
    noise_lvl = noises[noise_idx]
    pen = multiplier * noise_lvl

    poly, noised_signal, clean_signal, states, speeds = load_and_compute(
        datapath="../data",
        n_bkps=n_bkps,
        n_points=n_points,
        seed=seed,
        noise_idx=noise_idx,
        pos_heuristic="truth",
        speed_heuristic="truth",
        nstates=1,
    )
    #####################################
    #####################################
    # TREND FILTERING
    #####################################
    #####################################

    ytf = trend_filtering_pwq(signal=noised_signal.reshape(n_points, 1), vlambda=pen)[0]
    #####################################
    #####################################
    # Smoothing splines
    #####################################
    #####################################
    time_array = np.linspace(
        0, 1, num=n_points
    )  # set the interval over which to fit the spline
    tck, _, _, _ = splrep(
        x=time_array, y=noised_signal, k=2, task=0, s=pen, full_output=1
    )
    smsp_poly = interpolate.BSpline(*tck)

    #####################################
    #####################################
    # SplineOP
    #####################################
    #####################################

    cost = sop.cost_fn()
    model = sop.splineOP(cost)

    model.fit(noised_signal, states, speeds, True)
    model.predict(pen)

    #####################################
    #####################################
    # Build SPOP polynomial
    #####################################
    #####################################
    intbkps = model.bkps.astype(int)
    step_size = 1 / n_points
    n_steps = np.diff(intbkps)
    L = len(intbkps) - 1

    segment_start_speed = model.speed_path_mat[0, model.state_idx_sequence[0]]
    speed = np.array([segment_start_speed])
    acc = np.array([])

    # Rest of the points
    for bkp_idx in range(1, L):
        segment_start_speed = model.speed_path_mat[
            intbkps[bkp_idx], model.state_idx_sequence[bkp_idx]
        ]
        speed = np.append(
            arr=speed,
            values=segment_start_speed,
        )
        prev_seg_acc = np.array(
            [
                (speed[bkp_idx] - speed[bkp_idx - 1])
                / (2 * n_steps[bkp_idx - 1] * step_size)
            ]
        )
        acc = np.concatenate((acc, prev_seg_acc))

    final_speed = model.speed_path_mat[intbkps[L], model.state_idx_sequence[L]]
    prev_seg_acc = np.array(
        [(final_speed - speed[L - 1]) / (2 * n_steps[L - 1] * step_size)]
    )
    acc = np.concatenate((acc, prev_seg_acc))
    c = np.array([acc, speed, model.states[model.state_idx_sequence[:-1]]])
    x = intbkps

    spop_poly = interpolate.PPoly(c=c, x=x * step_size)  # spop poly

    #########################################
    ## Number of ruptures for each algorithm:
    cpt = np.where(~np.isclose(np.diff(ytf, n=3), 0, atol=1e-4))[0]
    cpt = np.append(arr=cpt, values=n_points)
    nbkps_tf = len(cpt) - 1

    nbkps_spop = len(model.bkps[1:-1])
    nbkps_smsp = len(np.unique(tck[0])[1:-1])
    xpoints = np.linspace(0, 1, n_points)

    yspop = spop_poly(xpoints)
    ysmsp = smsp_poly(xpoints)

    loss_spop = np.mean((yspop - clean_signal) ** 2), np.mean(
        (yspop - noised_signal) ** 2
    )
    loss_smsp = np.mean((ysmsp - clean_signal) ** 2), np.mean(
        (ysmsp - noised_signal) ** 2
    )
    loss_tf = np.mean((ytf - clean_signal) ** 2), np.mean((ytf - noised_signal) ** 2)

    f, ax = plt.subplots(figsize=(10, 10))
    xline = np.linspace(0, 1, 10000)

    ax.hlines(y=model.states, xmin=0, xmax=1, label="states", color="black")

    ax.plot(
        xline,
        spop_poly(xline),
        color="blue",
        lw=1,
        zorder=0,
        label=f"spop - nbkps {nbkps_spop} - MSE {loss_spop[0]:1.3e} - EMP_MSE {loss_spop[1]:1.3e}",
    )
    ax.plot(
        xpoints,
        ytf,
        color="lime",
        label=f"tf - nbkps {nbkps_tf} - MSE {loss_tf[0]:1.3e} - EMP_MSE {loss_tf[1]:1.3e}",
        lw=1,
    )
    ax.plot(
        xline,
        smsp_poly(xline),
        color="magenta",
        label=f"smsp - nbkps {nbkps_smsp}- MSE {loss_smsp[0]:1.3e} - EMP_MSE {loss_smsp[1]:1.3e}",
        lw=0.5,
    )

    ax.plot(
        xline,
        poly(xline),
        color="black",
        label=f"True nbkps: {n_bkps}",
        lw=1.5,
        ls="--",
    )
    ax.scatter(
        np.arange(0, n_points) / n_points, noised_signal, color="red", s=2, label="obs"
    )
    plt.title(
        f"n_points {n_points}, n_bkps {n_bkps}, seed {seed}, noise_idx {noise_idx}, pen {pen}, mult {round(multiplier)}"
    )
    plt.legend()
    ax.legend(bbox_to_anchor=(1.05, 0.5))


def compare_tf_spop(n_points, n_bkps, seed, noise_idx, tftol, multiplier, savedir=None):
    """
    n_points, n_bkps, seed, noise_idx to identify the signal.
    pen(alty) to run the models.
    """
    noises = {0: 0.1, 1: 0.13, 2: 0.16}
    noise_lvl = noises[noise_idx]
    pen = multiplier * noise_lvl

    poly, noised_signal, clean_signal, states, speeds = load_and_compute(
        datapath="../data",
        n_bkps=n_bkps,
        n_points=n_points,
        seed=seed,
        noise_idx=noise_idx,
        pos_heuristic="truth",
        speed_heuristic="truth",
        nstates=1,
    )
    #####################################
    #####################################
    # TREND FILTERING
    #####################################
    #####################################

    ytf = trend_filtering_pwq(signal=noised_signal.reshape(n_points, 1), vlambda=pen)[0]

    cost = sop.cost_fn()
    model = sop.splineOP(cost)

    model.fit(noised_signal, states, speeds, True)
    model.predict(pen)

    spop_poly = sop.get_polynomial_from_model(model)

    ## Number of ruptures for each algorithm:
    tf_cpt = np.where(~np.isclose(np.diff(ytf, n=3), 0, atol=tftol))[0]
    tf_cpt = np.unique(np.concatenate((np.array([0]), tf_cpt, np.array([n_points]))))
    tf_cpt = tf_cpt[1:-1]
    tf_spline = bkps_to_spline(yobs=noised_signal, bkps=tf_cpt / n_points, deg=2)
    # tf_cpt = np.append(arr=tf_cpt, values=n_points)
    nbkps_tf = len(tf_cpt)

    nbkps_spop = len(model.bkps[1:-1])

    ####################################
    # PLOT DE LOS PREDICHOS REALES     #
    ####################################
    f, ax = plt.subplots(figsize=(20, 10), ncols=2)

    ax_normal = ax[0]
    xline = np.linspace(0, 1, 10000)  # For plotting
    xpoints = np.linspace(0, 1, n_points)  # For error computations

    tf_mse = np.mean((ytf - clean_signal) ** 2)
    tf_empirical = np.mean((ytf - noised_signal) ** 2)

    yspop_plot = spop_poly(xline)
    yspop = spop_poly(xpoints)
    spop_spline = bkps_to_spline(yobs=noised_signal, bkps=spop_poly.x[1:-1], deg=2)

    spop_mse = np.mean((yspop - clean_signal) ** 2)
    spop_empirical = np.mean((yspop - noised_signal) ** 2)

    ax_normal.hlines(y=model.states, xmin=0, xmax=1, label="states", color="black")
    ymin = min(noised_signal)
    ymax = max(noised_signal)
    bkpstitle = ""

    if len(spop_poly.x) < 15:
        ax_normal.vlines(
            x=spop_poly.x[1:-1], ymin=ymin, ymax=ymax, lw=1, ls="--", color="blue"
        )
        bkpstitle = spop_poly.x

    ax_normal.plot(
        xline,
        yspop_plot,
        color="blue",
        lw=1,
        zorder=0,
        label=f"spop - nbkps {nbkps_spop} - MSE :{spop_mse:3.3E}- emp {spop_empirical:3.3E}",
    )
    ax_normal.plot(
        xpoints,
        ytf,
        color="limegreen",
        label=f"tf - nbkps {nbkps_tf}- MSE :{tf_mse:3.3E} - emp {tf_empirical:3.3E}",
        lw=1,
        ls=":",
    )

    ax_normal.plot(
        xline,
        poly(xline),
        color="black",
        label=f"True nbkps: {n_bkps}",
        lw=1.5,
        ls="--",
    )
    ax_normal.scatter(
        np.arange(0, n_points) / n_points, noised_signal, color="red", s=2, label="obs"
    )

    ax_normal.set_ylim((ymin - 0.01, ymax + 0.01))
    ax_normal.legend(fontsize="xx-large")
    ax_normal.legend(bbox_to_anchor=(0.5, -0.05))
    f.suptitle(
        f"n_points {n_points}, n_bkps {n_bkps}, seed {seed}, noise_idx {noise_idx}, pen {np.round(pen,3)}, mult {round(multiplier,3)}\n spop pred {bkpstitle[1:-1]}"
    )

    ####################################
    # PLOT DE LOS SPLINES CALCULADOS   #
    ####################################
    # fsplines, axsplines = plt.subplots()

    ytf_spline = tf_spline(xpoints)
    tfspline_mse = np.mean((ytf_spline - clean_signal) ** 2)
    tfspline_empirical_mse = np.mean((ytf_spline - noised_signal) ** 2)

    yspop_spline = spop_spline(xpoints)
    spopspline_mse = np.mean((yspop_spline - clean_signal) ** 2)
    spopspline_empirical_mse = np.mean((yspop_spline - noised_signal) ** 2)

    axsplines = ax[1]
    axsplines.plot(
        xline,
        spop_spline(xline),
        label=f"spop spline - MSE: {spopspline_mse:3.5E} - emp {spopspline_empirical_mse:3.5E}",
        color="blue",
    )
    axsplines.plot(
        xline,
        tf_spline(xline),
        ls="-.",
        label=f"tf spline - MSE: {tfspline_mse:3.5E} - emp {tfspline_empirical_mse:3.5E}",
        color="limegreen",
    )
    axsplines.plot(xline, poly(xline), label="f(x)", color="k", ls="--")

    if tfspline_mse < spopspline_mse:
        axsplines.annotate(text="X", xy=(0.1, 0.1), size="xx-large", weight="heavy")

    axsplines.scatter(
        np.unique(spop_spline.t)[1:-1],
        spop_spline(np.unique(spop_spline.t)[1:-1]),
        color="blue",
        s=100,
        zorder=3.5,
    )  # BSpline
    axsplines.scatter(
        tf_cpt / n_points, tf_spline(tf_cpt / n_points), color="limegreen", zorder=3.5
    )  # BSpline

    axsplines.scatter(
        poly.x[1:-1], poly(poly.x[1:-1]), color="k", marker="*", s=125, zorder=3.5
    )  # PPoly

    axsplines.scatter(
        np.arange(0, n_points) / n_points, noised_signal, color="red", s=1, label="obs"
    )
    yliminf = np.min(noised_signal) - 0.05
    ylimsup = np.max(noised_signal) + 0.05
    axsplines.set_ylim((yliminf, ylimsup))
    axsplines.legend(fontsize="xx-large")
    axsplines.legend(bbox_to_anchor=(0.5, -0.05))

    # print(f'{savedir}/{n_points}-{n_bkps}-{seed}-{noise_idx}-{multiplier}-tftol{int(-np.log10(tftol)):}')

    if savedir is not None:
        if savedir not in os.listdir():
            os.mkdir(savedir)
        f.savefig(
            f"{savedir}/{n_points}-{n_bkps}-{seed}-{noise_idx}-{multiplier}-tftol{int(-np.log10(tftol)):}.png",
            bbox_inches="tight",
        )
        plt.close()
    else:
        f.show()


def predict_from_bkps(row):

    n_bkps = row.true_n_bkps
    n_points = row.n_points
    seed = row.sample_seed
    noise_idx = row.noise_idx

    _, noised_signal, clean_signal, _, _ = load_and_compute(
        datapath="../data",
        n_bkps=n_bkps,
        n_points=n_points,
        seed=seed,
        noise_idx=noise_idx,
        pos_heuristic="truth",
        speed_heuristic="truth",
        nstates=1,
    )

    x = np.linspace(0, 1, num=n_points, endpoint=False)
    try:
        spline = bkps_to_spline(yobs=noised_signal, bkps=row.pred_bkps / row.n_points)
        yspline = spline(x)
        spline_mse = np.mean((yspline - clean_signal) ** 2)
    except:
        x = np.linspace(0, 1, num=n_points, endpoint=True)
        spline = bkps_to_spline(yobs=noised_signal, bkps=row.pred_bkps / row.n_points)
        yspline = spline(x)
        spline_mse = np.mean((yspline - clean_signal) ** 2)
    return spline_mse


HALL_DICTIONARY = {
    1: [0.7071, -0.7071],
    2: [0.8090, -0.5, -0.3090],
    3: [0.1942, 0.2809, 0.3832, -0.8582],
    4: [-0.2708, -0.0142, 0.6909, -0.4858, -0.4617],
    5: [0.9064, -0.2600, -0.2167, -0.1774, -0.1420, -0.1103],
    6: [0.2400, 0.0300, -0.0342, 0.07738, -0.3587, -0.3038, -0.3472],
    7: [0.9302, -0.1965, -0.1728, -0.1506, -0.1299, -0.1107, -0.0930, -0.0768],
    8: [0.2171, 0.0467, -0.0046, -0.0348, 0.8207, -0.2860, -0.2453, -0.2260, -0.2879],
    9: [
        0.9443,
        -0.1578,
        -0.1429,
        -0.1287,
        -0.1152,
        -0.1025,
        -0.0905,
        -0.0792,
        -0.0687,
        -0.0588,
    ],
    10: [
        0.1995,
        0.0539,
        0.0104,
        -0.0140,
        -0.0325,
        0.8510,
        -0.2384,
        -0.2079,
        -0.1882,
        -0.1830,
        -0.2507,
    ],
}


def sd_hall_diff(data):
    wei = np.array([0.1942, 0.2809, 0.3832, -0.8582])
    corrector = (
        wei[3] ** 2
        + (wei[2] - wei[3]) ** 2
        + (wei[1] - wei[2]) ** 2
        + (wei[0] - wei[1]) ** 2
        + wei[0] ** 2
    )

    z = np.diff(data)  # diff data
    n = len(z)
    mat = wei[:, None] @ z[:, None].T
    mat[1, : n - 1] = mat[1, 1:]
    mat[2, : n - 2] = mat[2, 2:]
    mat[3, : n - 3] = mat[3, 3:]
    sd = np.sqrt(np.sum(np.sum(mat[:, :-3], axis=0) ** 2) / ((n - 3) * corrector))
    return sd
