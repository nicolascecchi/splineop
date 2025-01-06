from __future__ import division
import sys
from splineop import splineop as spop
from splineop import sputils as spu
import itertools
import multiprocessing
import numpy as np
import json
from ruptures.metrics import precision_recall
from tqdm import tqdm
import csv
import os
import datetime
float_formatter = "{:.0f}".format
np.set_printoptions(formatter={'float_kind':float_formatter})




def f(inputs):
    # Get parameters from the input list
    (
        n_bkps,
        sample_seed,
        noise_idx,
        n_points,
        nstates,
        pos_heuristic,
        speed_heuristic,
        multiplier,
        K,
        save_folder,
    ) = inputs
    # Compute derived parameters from the inputs
    noise_dict = {0: 0.1, 1: 0.13, 2: 0.16}

    noise_lvl = noise_dict[noise_idx]

    # Get problem information
    poly, noised_signal, signal_clean, positions, speeds = spu.load_and_compute(
        datapath="data",
        n_bkps=n_bkps,
        n_points=n_points,
        seed=sample_seed,
        noise_idx=noise_idx,
        pos_heuristic=pos_heuristic,
        speed_heuristic=speed_heuristic,
        nstates=nstates,
    )
    # Still usefull for the case when using the true values
    # otherwise it should just equal the nstates argument
    nstates = len(positions)
    # Fit and predict
    model, run_time = spu.predict_pipeline(
        signal=noised_signal,
        positions=positions,
        speeds=speeds,
        K=K,
        normalized=True,
    )

    # Compute metrics
    total_error = np.min(model.soc[-1, -1])
    pred_n_bkps = len(model.bkps)
    spop_polynomial = spop.get_polynomial_from_constrained_model(model)
    
    x = np.linspace(start=0, stop=1, num=n_points, endpoint=False)
    ypred = spop_polynomial(x)
    total_quadratic_error = np.sum((ypred-signal_clean)**2)
    total_emp_error = np.sum((ypred-noised_signal)**2)

    true_bkps = np.round(poly.x[1:-1] * n_points)
    pred_breaks = model.bkps
    annotation_error = np.abs(pred_n_bkps - n_bkps)

    true_bkps_x = np.round(poly.x[1:] * n_points)  # for computations only
    pred_breaks_x = np.concat([model.bkps,np.array([model.n_points])])

    pr1, rc1 = precision_recall(true_bkps_x, pred_breaks_x, np.round(0.01 * n_points))
    pr25, rc25 = precision_recall(
        true_bkps_x, pred_breaks_x, np.round(0.025 * n_points)
    )
    pr5, rc5 = precision_recall(true_bkps_x, pred_breaks_x, np.round(0.05 * n_points))

    # Build results list with all items of interest
    results = {
        "n_bkps": n_bkps,  # 'n_bkp
        "n_points": n_points,
        "noise_idx": noise_idx,  # 'noise_idx'
        "noise": noise_lvl,
        "sample_seed": sample_seed,  # 'sample_seed'
        "pos_heuristic": pos_heuristic,  # 'pos_heur'
        "speed_heuristic": speed_heuristic,  # 'speed_heur'
        "nstates": nstates,  # 'nstates'
        "multiplier": multiplier,  # 'multiplier'
        "pred_n_bkps": pred_n_bkps,  # 'pred_n_bkps' the number of predicted breaks
        "pred_bkps": pred_breaks,  # 'pred_breaks' the position of the predicted breaks
        "true_n_bkps": len(true_bkps),
        "true_bkps": true_bkps,  # True break location
        "total_error": total_error,  # 'total_error'
        "emp_error": total_emp_error,
        "quadratic_error": total_quadratic_error,  # 'quadratic_error', total error - penalty errors
        "pr1": pr1,  # precision 1%
        "rc1": rc1,  # recall 1%
        "pr25": pr25,  # precision 2.5%
        "rc25": rc25,  # recall 2.5%
        "pr5": pr5,  # precision 5%
        "rc5": rc5,  # recall 5%
        "annotation_error": annotation_error,  # 'annotation_error'
        "algorithm": "spop",
        "time": run_time,
        "state_idx_seq":model.state_idx_sequence,
    }
    # Need to use this form of saving because of parallelism!
    with open(f"./{save_folder}/c-spop_results_{n_points}.csv", "a", newline="\n") as file:
        writer = csv.DictWriter(
            file, fieldnames=results.keys(), delimiter=";",dialect="excel",quoting=csv.QUOTE_MINIMAL
        )
        writer.writerow(results)


def main(n_points, save_folder):
    # assert len(sys.argv) == 3, f"Not enough arguments. 2 expected, received {len(sys.argv)-1}"
    # assert int(sys.argv[1]) in [500, 1000, 2000], f"n_points must be one of 500, 1000, 2000"

    n_points = [n_points]
    # save_folder = sys.argv[2]
    if save_folder not in os.listdir():
        os.mkdir(save_folder)
    n_bkps = range(1, 6)
    sample_seed = range(0, 50)
    noise_idx = [0, 1, 2]
    nstates = np.array([5, 10, 15, 20])
    nstates_placeholder = [
        1
    ]  # placeholder just for easy of computatiosn, it is ignored for non_heuristics.
    pos_non_heuristic = ["truth"]
    speed_heuristic = ["truth"]
    pos_heuristic = ["uniform", "qtiles"]
    #speed_heuristic = ["truth", "linreg"]  # ['linreg']#
    multiplier = np.logspace(
        start=-3, stop=4, num=16, endpoint=False
    )  # np.logspace(start=0,stop=4, num=15)
    K_range = np.arange(1,11,1)
    folder_as_list = [save_folder]

    print("WARNING NOTICE")
    print("Some important parameters are FIXED and NOT DISPLAYED here, inluding:")
    print(
        "-- Noise levels for the penalty that should coincide with the data generation values."
    )
    print(
        "-- Number of points considered for makingg the linear regression that estimates speed."
    )
    print("Parametrization")
    print("-----------------------------------------------------------" * 3)
    print(f"number of points: {n_points}")
    print(f"saving folder: {save_folder}")
    print(f"noise_idx: {noise_idx}")
    print(f"n_states: {nstates} ")
    print(f"position heuristic: {pos_heuristic}")
    print(f"speed heuristic: {speed_heuristic}")
    print(f"penalty multiplier: {multiplier} ")
    print(f"starting time: {datetime.datetime.today()}")
    print("-----------------------------------------------------------" * 3)

    heuristic_list = list(
        itertools.product(
            n_bkps,
            sample_seed,
            noise_idx,
            n_points,
            nstates,
            pos_heuristic,
            speed_heuristic,
            multiplier,
            K_range,
            folder_as_list,
        )
    )
    non_heuristic_list = list(
        itertools.product(
            n_bkps,
            sample_seed,
            noise_idx,
            n_points,
            nstates_placeholder,
            pos_non_heuristic,
            speed_heuristic,
            multiplier,
            K_range, 
            folder_as_list,
        )
    )

    final_running_list = non_heuristic_list #+ heuristic_list
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
        "state_idx_seq",
    ]

    with open(
        f"./{save_folder}/c-spop_results_{n_points[0]}.csv", "w+", newline="\n"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=fn, delimiter=";", dialect="excel",quoting=csv.QUOTE_MINIMAL)
        writer.writeheader()

    f(final_running_list[0])
    print(len(final_running_list))

    pool = multiprocessing.Pool(processes=32)
    num_tasks = len(final_running_list[:])
    for i, _ in enumerate(pool.imap(f, final_running_list, chunksize=8), 1):
        sys.stderr.write("\rdone {0:%}".format(i / num_tasks))

    pool.close()
