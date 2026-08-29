"""
P1 trainer wrapper: runs train_ladder.main() with build_optim swapped.

Usage (identical CLI to train_ladder.py, plus --arm):
  uv run python experiments/training/poc_stab/train_p1.py --arm polar \
      --dose 0.3 --steps 480 --tag p1_polar --mem-frac 0.55

The swap is the ONLY behavioral delta: data order, WSD schedule, QK-Clip,
clipping, logging, and seed discipline all come from train_ladder verbatim,
which is what makes the arms paired (plan doc P1, pre-registered).
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
POC_TWIN = os.path.join(os.path.dirname(HERE), "poc_twin")
LADDER = os.path.join(POC_TWIN, "ladder")
for p in (HERE, POC_TWIN, LADDER):
    if p not in sys.path:
        sys.path.insert(0, p)

import train_ladder as tl  # noqa: E402
from muon_hygiene import build_optim  # noqa: E402


def main():
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--arm", default="control",
                    choices=("control", "split", "polar", "nesterov"))
    arm_args, rest = ap.parse_known_args()
    tl.build_optim = (lambda model, lr, lr_embed, wd:
                      build_optim(model, lr, lr_embed, wd, arm=arm_args.arm))
    sys.argv = [sys.argv[0]] + rest
    print(f"[p1] arm={arm_args.arm} (build_optim swapped; rest verbatim)",
          flush=True)
    tl.main()


if __name__ == "__main__":
    main()
