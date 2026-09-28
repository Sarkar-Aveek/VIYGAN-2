"""
Everything in one command: collect ~100k L2A training tiles, train the ArcGIS-colour model (two runs), pick one,
train the Sentinel-colour model from it, and evaluate everything on held-out places.

Steps (every step can be re-run: downloads are cached, finished tiles and finished training runs are skipped,
an interrupted run resumes from its last checkpoint):
  arcgis      collectors/arcgis_collector.py --arcgis    ArcGIS targets around the training places (listed there)
  metadata    collectors/arcgis_collector.py --metadata  ArcGIS capture dates and resolution
  l2a         collectors/sentinel2_collector.py     8 clean L2A frames per tile, dates shared per block   (CDSE)
  build       collectors/build_training_data.py     filters, 100k tiles (42% cities), train/val split
  test_data   the same collectors with --test       held-out test places, in season and ~6 months off   (CDSE)
  train_A     train.py -opt configs/arcgis_A.yml    comparison run A: ArcGIS colours, the Satlas/India recipe
  train_B     train.py -opt configs/arcgis_B.yml    arcgis_B: ArcGIS colours, structure focus (GAN 0.05 + edge loss)
  pick        B if its val edge-F1 is >= 0.02 higher without LPIPS >= 0.01 worse, else A -> outputs/model1_choice.txt
  train_s2    configs/s2colour_stage1.yml, then configs/s2colour.yml
                                                    s2colour: Sentinel-2 colours (colour lock, colour-blind losses),
                                                    from arcgis_B
  evaluate    evaluate.py                           all models on the test places: CSVs, plots, mosaics

Before steps l2a and test_data download anything, a dry run with catalog searches (free) estimates the Sentinel Hub
processing units; the pipeline stops if a step would need more than --pu-budget.

    python pipeline.py --dry-run            # the plan
    python pipeline.py                      # everything
    python pipeline.py --from train_A       # data already collected
    python pipeline.py --only evaluate
    python pipeline.py --from train_A --retrain     # train the finished runs again
"""
import os
import re
import sys
import csv
import argparse
import subprocess

import bootstrap  # noqa: F401

STEPS = ["arcgis", "metadata", "l2a", "build", "test_data", "train_A", "train_B", "pick", "train_s2", "evaluate"]
NEEDS_CDSE = {"l2a", "test_data"}
EXP = "experiments"
CHOICE = os.path.join("outputs", "model1_choice.txt")
OFFSEASON_DAYS = 180
# pick rule: B replaces A only on a clear structure gain without a clear perceptual loss
MIN_EDGE_F1_GAIN = 0.02
MAX_LPIPS_LOSS = 0.01


def py(*args):
    return [sys.executable, *args]


def run(cmd):
    print(f"\n$ {' '.join(cmd[1:])}", flush=True)
    return subprocess.run(cmd).returncode


def expected_pu(cmd):
    """Runs a collector's --dry-run --catalog (0 PU) and returns its expected PU."""
    out = subprocess.run(cmd + ["--dry-run", "--catalog"], capture_output=True, text=True)
    print(out.stdout[-1500:], end="")
    if out.returncode != 0:
        sys.exit(f"Dry run failed:\n{out.stderr[-2000:]}")
    m = re.search(r"Processing units: expected ~(\d+), worst (\d+)", out.stdout)
    return int(m.group(1)) if m else 0


def collect_l2a(args, extra=()):
    cmd = py("collectors/sentinel2_collector.py", *extra)
    pu = expected_pu(cmd)
    print(f"{' '.join(cmd[1:])}: expected ~{pu} PU, budget {args.pu_budget}")
    if pu > args.pu_budget:
        sys.exit(f"Stopping: it would need more than --pu-budget {args.pu_budget} PU. Check the account's remaining "
                 "quota, then raise --pu-budget or lower the tile counts in collectors/arcgis_collector.py.")
    return run(cmd)


def finished(run_name):
    return os.path.exists(os.path.join(EXP, run_name, "models", "net_g_latest.pth"))


def train(run_name, config, args, overrides=()):
    if finished(run_name) and not args.retrain:
        print(f"\n{run_name}: already trained, skipped (--retrain to redo)")
        return 0
    cmd = py("train.py", "-opt", config)
    states = os.path.join(EXP, run_name, "training_states")
    if not args.retrain and os.path.isdir(states) and os.listdir(states):
        cmd.append("--auto_resume")
    if overrides:
        cmd += ["--force_yml", *overrides]
    return run(cmd)


def last_val(run_name):
    path = os.path.join(EXP, run_name, "val.csv")
    if not os.path.exists(path):
        return None
    rows = list(csv.DictReader(open(path)))
    return {k: float(v) for k, v in rows[-1].items() if k not in ("iter", "epoch")} if rows else None


def pick():
    a, b = last_val("arcgis_A"), last_val("arcgis_B")
    if a is None:
        print("pick: arcgis_A has no validation yet")
        return 1
    if b is None:
        choice, why = "arcgis_A", "arcgis_B has no validation"
    else:
        gain, loss = b["edge_f1"] - a["edge_f1"], b["lpips"] - a["lpips"]
        choice = "arcgis_B" if gain >= MIN_EDGE_F1_GAIN and loss < MAX_LPIPS_LOSS else "arcgis_A"
        why = (f"edge-F1 A {a['edge_f1']:.4f} B {b['edge_f1']:.4f} (B {gain:+.4f}), "
               f"LPIPS A {a['lpips']:.4f} B {b['lpips']:.4f} (B {loss:+.4f}); "
               f"rule: B needs >= +{MIN_EDGE_F1_GAIN} edge-F1 and < +{MAX_LPIPS_LOSS} LPIPS")
    os.makedirs(os.path.dirname(CHOICE), exist_ok=True)
    with open(CHOICE, "w") as f:
        f.write(choice + "\n")
    with open(CHOICE.replace(".txt", "_why.txt"), "w") as f:
        f.write(why + "\n")
    print(f"pick: {choice}  ({why})")
    return 0


def train_s2(args):
    """s2colour in two stages: stage 1 from arcgis_B (24k iterations, fresh high-pass discriminator), then stage 2
    from stage 1's 18k checkpoint (2 more epochs at the full learning rate)."""
    rc = train("s2colour_24k_v1", "configs/s2colour_stage1.yml", args)
    return rc or train("s2colour", "configs/s2colour.yml", args)


def commands(step, args):
    """Returns a function that runs the step and gives its exit code."""
    return {
        "arcgis": lambda: run(py("collectors/arcgis_collector.py", "--arcgis")),
        "metadata": lambda: run(py("collectors/arcgis_collector.py", "--metadata")),
        "l2a": lambda: collect_l2a(args),
        "build": lambda: run(py("collectors/build_training_data.py")),
        "test_data": lambda: (run(py("collectors/arcgis_collector.py", "--test"))
                              or collect_l2a(args, ["--test"])
                              or collect_l2a(args, ["--test", "--offseason", str(OFFSEASON_DAYS)])),
        "train_A": lambda: train("arcgis_A", "configs/arcgis_A.yml", args),
        "train_B": lambda: train("arcgis_B", "configs/arcgis_B.yml", args),
        "pick": pick,
        "train_s2": lambda: train_s2(args),
        "evaluate": lambda: run(py("evaluate.py")),
    }[step]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--from", dest="start", choices=STEPS, default=STEPS[0], help="start at this step")
    p.add_argument("--only", nargs="+", choices=STEPS, help="run only these steps")
    p.add_argument("--pu-budget", type=int, default=15000,
                   help="most Sentinel Hub processing units one download step may use")
    p.add_argument("--retrain", action="store_true", help="train runs again even if they finished")
    p.add_argument("--dry-run", action="store_true", help="print the steps and stop")
    args = p.parse_args()

    steps = args.only or STEPS[STEPS.index(args.start):]
    steps = [s for s in STEPS if s in steps]
    for s in steps:
        print(f"  {STEPS.index(s) + 1:2d} {s}")
    if args.dry_run:
        return
    if NEEDS_CDSE & set(steps) and not (os.environ.get("CDSE_CLIENT_ID") and os.environ.get("CDSE_CLIENT_SECRET")):
        sys.exit("Set CDSE_CLIENT_ID and CDSE_CLIENT_SECRET in the environment or in .env (see .env.example).")

    for s in steps:
        code = commands(s, args)()
        if code:
            sys.exit(f"\nStep '{s}' did not finish (exit code {code}). Fix the cause and run:\n"
                     f"  python pipeline.py --from {s}")
    print("\nDone. Results: outputs/eval/ (summary.csv, metrics.png, tradeoff.png, by_kind.png, curves_*.png, sheets/)")


if __name__ == "__main__":
    main()
