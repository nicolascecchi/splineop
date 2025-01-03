import cvxpy as cp
from scipy.interpolate import make_smoothing_spline
import sys
from time import process_time
import numpy as np
import itertools
import csv
from splineop import sputils as spu
from ruptures.metrics import precision_recall
import multiprocessing
import os
import datetime

float_formatter = "{:.0f}".format
np.set_printoptions(formatter={'float_kind':float_formatter})


def trend_filtering_pwq(signal: np.ndarray, vlambda: float):
    degree = 2
    u = cp.Variable(shape=signal.shape, name="u")
    obj = cp.Minimize(
        cp.sum_squares(signal - u)
        + vlambda * cp.mixed_norm(cp.diff(u, k=degree + 1), p=2, q=1)
    )
    prob = cp.Problem(obj)
    prob.solve(solver=cp.CLARABEL, verbose=False)  # modify solver if too long
    if prob.status != cp.OPTIMAL:
        raise Exception("Solver did not converge!")
    signal_smoothed = u.value
    return signal_smoothed


def f(inputs):
    # Get parameters from the input list
    n_bkps, sample_seed, noise_idx, n_points, multiplier, save_folder = inputs
    # Compute derived parameters from the inputs
    noise_dict = {0: 0.1, 1: 0.13, 2: 0.16}

    noise_lvl = noise_dict[noise_idx]
    pen = multiplier * noise_lvl

    # Get problem information
    poly, noised_signal, clean_signal = spu.load_and_compute(
        datapath="data",
        n_bkps=n_bkps,
        n_points=n_points,
        seed=sample_seed,
        noise_idx=noise_idx,
        pos_heuristic=None,
        speed_heuristic=None,
        nstates=None,
    )

    noised_signal = np.expand_dims(noised_signal, axis=1)  # For trend filtering only
    clean_signal = np.expand_dims(clean_signal, axis=1)
    # Put this here before the TRY statement, so in case of failure we have the values anyway
    true_bkps = np.round(poly.x[1:-1] * n_points)
    true_bkps_x = np.round(poly.x[1:] * n_points)  # xtended for computation only

    # Try, except block because the optimization algo may not converge
    try:
        t0 = process_time()
        yy = trend_filtering_pwq(signal=noised_signal, vlambda=pen)
        run_time = process_time() - t0
        yy = yy.flatten()
        #print(yy.shape)
        # Compute metrics
        cpt = np.where(~np.isclose(np.diff(yy, n=3), 0, atol=1e-4))[0]
        cpt = np.unique(
            np.concatenate((np.array([0]), cpt, np.array([n_points])))
        )  # artificially add 0 and n_points, but remove possible duplicate of 0
        
        yy = yy.reshape(clean_signal.shape) # Reshape so that 
        pred_n_bkps = len(cpt) - 2  # discount 0 and n_points
        total_quadratic_error = np.sum((yy - clean_signal) ** 2)
        total_emp_error = np.sum((yy-noised_signal)**2)
        total_error = total_emp_error + pen * pred_n_bkps

        pred_bkps = cpt[1:-1]  # remove 0 and n_points
        pred_bkps_x = cpt[1:]  # xtended for computation only
        annotation_error = np.abs(pred_n_bkps - n_bkps)

        pr1, rc1 = precision_recall(true_bkps_x, pred_bkps_x, np.round(0.01 * n_points))
        pr25, rc25 = precision_recall(
            true_bkps_x, pred_bkps_x, np.round(0.025 * n_points)
        )
        pr5, rc5 = precision_recall(true_bkps_x, pred_bkps_x, np.round(0.05 * n_points))
        # Build results list with all items of interest
        results = {
            "n_bkps": n_bkps,  # 'n_bkp
            "n_points": n_points,
            "noise_idx": noise_idx,  # 'noise_idx'
            "noise": noise_lvl,
            "sample_seed": sample_seed,  # 'sample_seed'
            "pos_heuristic": np.nan,  # 'pos_heur'
            "speed_heuristic": np.nan,  # 'speed_heur'
            "nstates": np.nan,  # 'nstates'
            "multiplier": multiplier,  # 'multiplier'
            "pred_n_bkps": pred_n_bkps,  # 'pred_n_bkps' the number of predicted breaks
            "pred_bkps": pred_bkps,  # 'pred_bkps' the position of the predicted breaks
            "true_n_bkps": len(true_bkps),
            "true_bkps": true_bkps,  # True break location
            "total_error": total_error,  # 'total_error'
            "emp_error": total_emp_error,
            "quadratic_error": total_quadratic_error,  # 'quadratic_error', total error - penalty errors
            "pr1": pr1,
            "rc1": rc1,
            "pr25": pr25,
            "rc25": rc25,
            "pr5": pr5,
            "rc5": rc5,
            "annotation_error": annotation_error,  # 'annotation_error'
            "algorithm": "tf",
            "time": run_time,
        }
    except Exception as err:
        print(err)
        results = {
            "n_bkps": n_bkps,  # 'n_bkp
            "n_points": n_points,
            "noise_idx": noise_idx,  # 'noise_idx'
            "noise": noise_lvl,
            "sample_seed": sample_seed,  # 'sample_seed'
            "pos_heuristic": np.nan,  # 'pos_heur'
            "speed_heuristic": np.nan,  # 'speed_heur'
            "nstates": np.nan,  # 'nstates'
            "multiplier": multiplier,  # 'multiplier'
            "pred_n_bkps": np.nan,  # 'pred_n_bkps' the number of predicted breaks
            "pred_bkps": np.nan,  # 'pred_bkps' the position of the predicted breaks
            "true_n_bkps": len(true_bkps),
            "true_bkps": true_bkps,  # True break location
            "total_error": np.nan,  # 'total_error'
            "emp_error": np.nan,
            "quadratic_error": np.nan,  # 'quadratic_error', total error - penalty errors
            "pr1": np.nan,  # precision 1%
            "rc1": np.nan,  # recall 1%
            "pr25": np.nan,  # precision 2.5%
            "rc25": np.nan,  # recall 2.5%
            "pr5": np.nan,  # precision 5%
            "rc5": np.nan,  # recall 5%
            "annotation_error": np.nan,  # 'annotation_error'
            "algorithm": "tf",
            "time": np.nan,
        }
    finally:
        # Need to use this form of saving because of parallelism!
        with open(
            f"./{save_folder}/tf_results_{n_points}.csv", "a", newline="\n"
        ) as file:
            writer = csv.DictWriter(
                file, fieldnames=results.keys(), delimiter=";", dialect="excel",quoting=csv.QUOTE_MINIMAL
            )
            writer.writerow(results)


def main(n_points, save_folder):
    # assert len(sys.argv) == 3, f"Not enough arguments. 2 expected, received {len(sys.argv)-1}"
    # assert int(sys.argv[1]) in [500, 1000, 2000], f"n_points must be one of 500, 1000, 2000"

    # Read input
    n_points = [n_points]
    if save_folder not in os.listdir():
        os.mkdir(save_folder)

    # Set variables
    n_bkps = range(1, 6)
    sample_seed = range(0, 50)
    noise_idx = [0, 1, 2]

    multiplier = np.logspace(
        start=-3, stop=4, num=16, endpoint=False
    )  # np.logspace(start=0,stop=4, num=15)
    folder_as_list = [save_folder]
    # Set variable lists
    final_running_list = list(
        itertools.product(
            n_bkps, sample_seed, noise_idx, n_points, multiplier, folder_as_list
        )
    )

    print("Parametrization")
    print("-----------------------------------------------------------" * 3)
    print(
        f"number of points: {n_points} \n saving folder: {save_folder} \n noise_idx: {noise_idx} \n penalty multiplier: {multiplier} \n starting time: {datetime.datetime.today()}"
    )
    print("-----------------------------------------------------------" * 3)

    field_names = [
        "n_bkps",
        "n_points",
        "noise_idx",
        "noise",
        "sample_seed",
        "pos_heuristic",
        "speed_heuristic",
        "nstates",
        "multiplier",
        "pred_n_bkps",
        "pred_bkps",
        "true_n_bkps",
        "true_bkps",
        "total_error",
        "emp_error",
        "quadratic_error",
        "pr1",
        "rc1",
        "pr25",
        "rc25",
        "pr5",
        "rc5",
        "annotation_error",
        "algorithm",
        "time",
    ]

    # Create file for saving results
    with open(f"./{save_folder}/tf_results_{n_points[0]}.csv", "w", newline="\n") as file:
        writer = csv.DictWriter(
            file, fieldnames=field_names, delimiter=";", dialect="excel",quoting=csv.QUOTE_MINIMAL
        )
        writer.writeheader()

    # First run
    # Uses this to make the first numba run that compiles and keeps the fast version in cache
    # It is adding an artificial first run whose results are valid but the time measurement is not
    # accurate since it will count compilation time
    # Hence we afterwards re-run this configuration
    f(final_running_list[0])
    print(len(final_running_list))  # just to get the feeling

    # Run all the experiences
    pool = multiprocessing.Pool(32)
    num_tasks = len(final_running_list[:])  # to print % advancement on shell
    for i, _ in enumerate(pool.imap(f, final_running_list, chunksize=8), 1):
        sys.stderr.write("\rdone {0:%}".format(i / num_tasks))

    pool.close()
