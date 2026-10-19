import argparse
import subprocess
import sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description="Repeat the calibrated pouring acquisition")
    p.add_argument("folder")
    p.add_argument("begin", type=int)
    p.add_argument("end", type=int)
    p.add_argument("--config", default="hardware/config.yaml")
    args = p.parse_args()
    if args.begin < 1 or args.end < args.begin:
        p.error("Require 1 <= begin <= end")
    root = Path(args.folder)
    root.mkdir(parents=True, exist_ok=True)
    for i in range(args.begin, args.end + 1):
        subprocess.run(
            [
                sys.executable,
                "-m",
                "hardware.acquisition.record_trial",
                str(root / f"event{i}.raw"),
                str(root / f"rgb{i}"),
                "--config",
                args.config,
            ],
            check=True,
        )


if __name__ == "__main__":
    main()
