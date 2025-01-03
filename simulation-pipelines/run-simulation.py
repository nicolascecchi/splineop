import sys

sys.path.append("../")
sys.path.append(".")
import argparse
import os
import datetime

parser = argparse.ArgumentParser()
parser.add_argument("algo", help="Algorithm to run simulation. One of [sop, tf, smsp]")
parser.add_argument("npoints", help="Number of points in the signal to load.", type=int)
parser.add_argument(
    "--savefolder", help="Directory where to save the results.", default=None
)
args = parser.parse_args()

if args.savefolder is None:
    savefolder = (
        str(datetime.datetime.today())
        .replace(" ", "-")
        .replace(":", "-")
        .replace(".", "-")
    )
    savefolder = args.algo + "-" + str(args.npoints) + "-" + savefolder
    args.savefolder = savefolder
if __name__ == "__main__":
    match args.algo:
        case "sop":
            import runSplineop

            runSplineop.main(args.npoints, args.savefolder)
        case "tf":
            import runTrendFiltering

            runTrendFiltering.main(args.npoints, args.savefolder)
        case "smsp":
            import runSmoothingSpline

            runSmoothingSpline.main(args.npoints, args.savefolder)
        case _:
            raise ValueError("Unsupported algorithm. Check spelling. ")
