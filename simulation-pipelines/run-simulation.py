import sys

sys.path.append("../")
sys.path.append(".")
import argparse
import os
import datetime

parser = argparse.ArgumentParser()
parser.add_argument("--algo", help="Algorithm to run simulation. One of [c-spop, p-spop, tf, smsp]")
parser.add_argument("--npoints", help="Number of points in the signal to load.", type=int)
parser.add_argument(
    "--savefolder", help="Directory where to save the results.", default=None
)
parser.add_argument(
    "--heuristic", help="Heuristic for positions. One of ['truth', 'quantiles', 'both']",default='both' 
    )
parser.add_argument("--max-signals", help="Nb of signals.", default=50, type=int)
parser.add_argument("--signal-n-bkps",help="Nb of changes in signal.", nargs='*', default=[1,2,3,4,5], type=int)
parser.add_argument("--noise",help="Noise levels.", nargs='*', default=[0,1,2], type=int)

args = parser.parse_args()

if args.savefolder is None:
    savefolder = (
        str(datetime.datetime.today())
        .replace(" ", "-")
        .replace(":", "-")[0:16]
    )
    savefolder = 'results/' + args.algo + "-" + str(args.npoints) + "-" + savefolder
    args.savefolder = savefolder
if __name__ == "__main__":
    match args.algo:
        case "c-spop":
            import runConstrainedSplineop
            runConstrainedSplineop.main(args.npoints, args.savefolder, args.heuristic, args.max_signals, args.signal_n_bkps, args.noise)
        
        case "p-spop":
            import runPenalizedSplineop
            runPenalizedSplineop.main(args.npoints, args.savefolder, args.heuristic, args.max_signals, args.signal_n_bkps, args.noise)
        
        case "tf":
            import runTrendFiltering
            runTrendFiltering.main(args.npoints, args.savefolder, args.heuristic, args.max_signals, args.signal_n_bkps, args.noise)
        
        case "smsp":
            import runSmoothingSpline
            runSmoothingSpline.main(args.npoints, args.savefolder, args.heuristic, args.max_signals, args.signal_n_bkps, args.noise)
        
        case _:
            raise ValueError("Unsupported algorithm. Check spelling. ")
