import cvxpy as cp
from scipy.interpolate import make_smoothing_spline
import sys
from time import process_time
import numpy as np
import itertools
import csv
from splineop import splineop as sop
from splineop import sputils as spu
from ruptures.metrics import precision_recall
import multiprocessing
from scipy.interpolate import PPoly, make_smoothing_spline, BSpline, splrep, splev
import os
import datetime

float_formatter = "{:.0f}".format
np.set_printoptions(formatter={'float_kind':float_formatter})

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
    # Fit and predict
    time_array = np.linspace(start=0, stop=1, num=n_points, endpoint=True)
    t0 = process_time()
    tck, error, _, _ = splrep(x=time_array, y=noised_signal, k=2, task=0, s=pen, full_output=1)
    run_time = process_time() - t0

    cpt = np.unique(tck[0])
    pred_bkps_x = np.round(
        cpt * n_points
    )  # Normally should include extremes, but just in case
    pred_bkps_x = np.unique(
        np.concatenate((np.array([0]), pred_bkps_x, np.array([n_points])))
    )

    # Compute metrics
    smsp_spline = BSpline(*tck)
    ypred = smsp_spline(time_array)
    total_quadratic_error = np.sum((ypred - clean_signal)**2)
    total_emp_error = np.sum((ypred-noised_signal)**2)

    pred_n_bkps = len(pred_bkps_x) - 2
    total_error = total_emp_error + pen * pred_n_bkps

    pred_bkps = pred_bkps_x[1:-1]
    annotation_error = np.abs(pred_n_bkps - n_bkps)

    true_bkps = np.round(poly.x[1:-1] * n_points)
    true_bkps_x = np.round(poly.x[1:] * n_points)

    pr1, rc1 = precision_recall(true_bkps_x, pred_bkps_x, np.round(0.01 * n_points))
    pr25, rc25 = precision_recall(true_bkps_x, pred_bkps_x, np.round(0.025 * n_points))
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
        "pred_n_bkps": pred_n_bkps,  # 'pred_n_bkps' the number of predicted bkps
        "pred_bkps": pred_bkps,  # 'pred_bkps' the position of the predicted bkps
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
        "algorithm": "smsp",
        "time": run_time,
    }
    # Need to use this form of saving because of parallelism!
    with open(f"./{save_folder}/smsp_results_{n_points}.csv", "a", newline="\n") as file:
        writer = csv.DictWriter(
            file, fieldnames=results.keys(), delimiter=";", dialect="excel",quoting=csv.QUOTE_MINIMAL,
        )
        writer.writerow(results)


def main(n_points, save_folder, heuristic, max_signals, signal_n_bkps):
    # assert len(sys.argv) == 3, f"Not enough arguments. Expected 2, received {len(sys.argv)-1}"
    # assert int(sys.argv[1]) in [500, 1000, 2000], f"n_points must be one of 500, 1000, 2000"
    # Read input
    n_points = [n_points]
    if save_folder not in os.listdir():
        os.mkdir(save_folder)
    # Signals to analyze
    n_bkps = signal_n_bkps
    sample_seed = range(0, max_signals) # (0,50)
    noise_idx = noise

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

    fn = [
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
    with open(
        f"./{save_folder}/smsp_results_{n_points[0]}.csv", "w", newline="\n"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=fn, delimiter=";", dialect="excel",quoting=csv.QUOTE_MINIMAL)
        writer.writeheader()

    # First run
    f(final_running_list[0])
    print(len(final_running_list))
    # Run all the experiences
    pool = multiprocessing.Pool(70)
    num_tasks = len(final_running_list[:])
    for i, _ in enumerate(pool.imap(f, final_running_list, chunksize=8), 1):
        sys.stderr.write("\rdone {0:%}".format(i / num_tasks))

    pool.close()
