import argparse
import pandas as pd
from .plotting import plot_lines_by_key, plot_heatmap_at_step
import matplotlib.pyplot as plt

def main():
    p = argparse.ArgumentParser("gradviz")
    sub = p.add_subparsers(dest="cmd", required=True)

    pl = sub.add_parser("plot", help="Line-plot gradients over steps")
    pl.add_argument("csv", help="Path to gradviz CSV")
    pl.add_argument("--by", choices=["layer","param"], default="layer")
    pl.add_argument("--topk", type=int, default=20)
    pl.add_argument("--save", default=None)
    pl.add_argument("--noshow", action="store_true")

    hm = sub.add_parser("heatmap", help="Heatmap at a step")
    hm.add_argument("csv", help="Path to gradviz CSV")
    hm.add_argument("--by", choices=["layer","param"], default="layer")
    hm.add_argument("--step", type=int, default=None)
    hm.add_argument("--save", default=None)
    hm.add_argument("--noshow", action="store_true")

    args = p.parse_args()
    df = pd.read_csv(args.csv)

    if args.cmd == "plot":
        plot_lines_by_key(df, by=args.by, topk=args.topk)
        if args.save:
            plt.savefig(args.save, bbox_inches="tight", dpi=200)
        if not args.noshow:
            plt.show()
    elif args.cmd == "heatmap":
        plot_heatmap_at_step(df, by=args.by, at_step=args.step)
        if args.save:
            plt.savefig(args.save, bbox_inches="tight", dpi=200)
        if not args.noshow:
            plt.show()

if __name__ == "__main__":
    main()
