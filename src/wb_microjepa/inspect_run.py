import argparse
from pathlib import Path
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=str)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    metrics = pd.read_csv(run_dir / "metrics.csv")
    stats = pd.read_csv(run_dir / "microbrain_stats.csv")

    print("\n=== Latest metrics ===")
    print(metrics.tail(10).to_string(index=False))

    print("\n=== Top active micro-brains on probe set ===")
    print(
        stats.sort_values("mean_probe_activation", ascending=False)
        .head(20)
        .to_string(index=False)
    )

    print("\nPlots:")
    for p in sorted((run_dir / "plots").glob("*.png"))[-10:]:
        print(" -", p)


if __name__ == "__main__":
    main()
