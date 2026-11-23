"""Run the steps of the FLAME pipeline in order.

python run_pipeline.py                     # all steps
python run_pipeline.py --from 05 --to 07   # a range of steps
python run_pipeline.py --skip 04a 04b      # e.g. reuse the trained audio model
"""

import argparse
import subprocess
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent / "scripts"
PIPELINE_DIR = SCRIPTS / "flame_pipeline"


def step_id(script: Path) -> str:
    return script.name.split("_")[0]


def select_steps(first, last, skip):
    return [
        s
        for s in sorted(PIPELINE_DIR.glob("[0-9]*.py"))
        if (first is None or step_id(s) >= first)
        and (last is None or step_id(s) <= last)
        and step_id(s) not in skip
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--from", dest="first", help="first step id, e.g. 02")
    parser.add_argument("--to", dest="last", help="last step id, e.g. 05")
    parser.add_argument("--skip", nargs="*", default=[], help="step ids to skip, e.g. 04a 04b")
    args = parser.parse_args()

    for script in select_steps(args.first, args.last, set(args.skip)):
        print(f"\n===== {script.relative_to(SCRIPTS)} =====", flush=True)
        subprocess.run([sys.executable, str(script)], check=True)
    print("\nDone. Results are in outputs/.")


if __name__ == "__main__":
    main()
