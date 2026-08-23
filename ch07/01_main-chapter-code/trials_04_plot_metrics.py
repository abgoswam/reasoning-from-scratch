# Concept: plotting GRPO metrics (notebook cells 68-73, section 7.3.3)
#
# Two small functions do all the work: moving_average smooths a very noisy
# signal, and plot_grpo_metrics lays four columns out on one grid so you can
# see them move together. Both are worth reading -- the smoothing in
# particular can hide the thing you are looking for.
#
# Reads ch07/02_logs/ and writes PNGs. No model, no GPU.
#
#   Part 1: moving_average by hand
#   Part 2: what window_fraction=0.25 actually does
#   Part 3: what smoothing hides
#   Part 4: render the reference runs

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LOGS = Path(__file__).resolve().parent.parent / "02_logs"
OUT = Path(__file__).resolve().parent / "plots"


def moving_average(values, window_fraction=0.25):
    """Identical to reasoning_from_scratch/ch07.py -- copied so this file stands alone."""
    window_size = max(1, int(window_fraction * len(values)))
    smoothed = []
    for i in range(len(values)):
        start_idx = max(0, i - window_size + 1)
        window_mean = sum(values[start_idx:i + 1]) / (i - start_idx + 1)
        smoothed.append(window_mean)
    return smoothed


def read_col(name, column):
    with (LOGS / name).open(newline="") as f:
        return [float(r[column]) for r in csv.DictReader(f)
                if r.get("step") and r.get(column)]


# ---------------------------------------------------------------------------
# Part 1 -- it is a TRAILING mean, not a centered one.
#
# Each output looks back at most window_size samples and never forward, so
# the smoothed curve lags the real one. That matters when you are trying to
# pin down the step where something broke.
# ---------------------------------------------------------------------------
def part1_by_hand():
    print("=" * 72)
    print("PART 1  moving_average is a trailing mean")
    print("=" * 72)

    values = [0., 0., 0., 0., 10., 0., 0., 0.]
    smoothed = moving_average(values, window_fraction=0.5)   # window = 4

    print(f"  input   : {values}")
    print(f"  window  : {max(1, int(0.5 * len(values)))}")
    print(f"  smoothed: {[round(v, 3) for v in smoothed]}\n")
    print(f"  {'i':>3} {'value':>7} {'smoothed':>10}  window used")
    print("  " + "-" * 46)
    w = max(1, int(0.5 * len(values)))
    for i, (v, sm) in enumerate(zip(values, smoothed)):
        start = max(0, i - w + 1)
        print(f"  {i:>3} {v:>7.1f} {sm:>10.3f}  values[{start}:{i+1}]")

    print("\n  -> the spike at i=4 is still visible at i=5,6,7. The smoothed")
    print("     curve says 'something happened recently', not 'here'.")
    print("     Early points average over fewer samples, so the curve starts")
    print("     tight to the data and loosens as it goes.\n")


# ---------------------------------------------------------------------------
# Part 2 -- the window scales with the run.
#
# window_fraction is a FRACTION, so a 500-step run smooths over 125 steps
# while a 10-step run smooths over 2. The same call behaves very differently
# depending on how long you trained.
# ---------------------------------------------------------------------------
def part2_window_scales():
    print("=" * 72)
    print("PART 2  the window scales with run length")
    print("=" * 72)

    print(f"  {'run length':>12} {'window @0.25':>14} {'window @0.05':>14}")
    print("  " + "-" * 44)
    for n in (10, 50, 100, 500, 2000):
        print(f"  {n:>12} {max(1, int(0.25 * n)):>14} "
              f"{max(1, int(0.05 * n)):>14}")

    print("\n  -> on your 10-step run the default window is 2, so the 'smoothed'")
    print("     line is barely smoothed. On a 500-step run it is 125, which")
    print("     flattens anything shorter than ~100 steps into nothing.\n")


# ---------------------------------------------------------------------------
# Part 3 -- what a quarter-length window hides.
#
# The 7.5 KL run collapses to 0% accuracy. Its reward signal goes flat. At
# window_fraction=0.25 the transition is smeared over 125 steps; at 0.02 you
# can see roughly where it went.
# ---------------------------------------------------------------------------
def part3_what_smoothing_hides():
    print("=" * 72)
    print("PART 3  smoothing can bury the moment it broke")
    print("=" * 72)

    name = "7_5_plus_kl_metrics.csv"
    if not (LOGS / name).exists():
        print(f"  {name} not found\n")
        return

    reward = read_col(name, "reward_avg")
    print(f"  7.5 + KL, {len(reward)} steps of reward_avg\n")

    def first_flatline(values, tol=1e-9, run=25):
        streak = 0
        for i, v in enumerate(values):
            streak = streak + 1 if abs(v) < tol else 0
            if streak >= run:
                return i - run + 1
        return None

    raw_at = first_flatline(reward)
    print(f"  raw signal goes permanently flat around step : {raw_at}")
    for wf in (0.02, 0.10, 0.25):
        sm = moving_average(reward, window_fraction=wf)
        at = first_flatline(sm, tol=1e-6)
        w = max(1, int(wf * len(reward)))
        print(f"  window_fraction={wf:<5} (window={w:>3}) reports  : {at}")

    print("\n  -> larger windows report the break LATER, by roughly the window")
    print("     length. Plot smoothed for the trend, but go back to the raw")
    print("     CSV when you want the step number.\n")


# ---------------------------------------------------------------------------
# Part 4 -- the actual figures.
#
# plot_grpo_metrics puts four columns on a 2x2 grid sharing an x-axis, draws
# the raw series faint and the moving average solid, and uses BARS for
# eval_acc because it is measured only at checkpoints -- a line would imply
# values that were never taken.
# ---------------------------------------------------------------------------
def plot_grpo_metrics(csv_path, columns, save_as=None):
    data = {name: {"steps": [], "values": []} for name in columns}
    with Path(csv_path).open(newline="") as f:
        for row in csv.DictReader(f):
            if not row or not row.get("step"):
                continue
            step = int(row["step"])
            for name in columns:
                if row.get(name):
                    data[name]["steps"].append(step)
                    data[name]["values"].append(float(row[name]))

    fig, axes = plt.subplots(2, 2, sharex=True, figsize=(9, 6))
    axes = axes.ravel()
    for i, name in enumerate(columns):
        steps, values = data[name]["steps"], data[name]["values"]
        if not values:
            fig.delaxes(axes[i])
            continue
        if name == "eval_acc":
            axes[i].bar(steps, values, width=20)
        else:
            axes[i].plot(steps, values, alpha=0.4)
            axes[i].plot(steps, moving_average(values))
        axes[i].set_ylabel(name)
    for j in (2, 3):
        if axes[j] in fig.axes:
            axes[j].set_xlabel("Step")
    fig.tight_layout()
    if save_as:
        fig.savefig(save_as, dpi=110)
    plt.close(fig)


def part4_render():
    print("=" * 72)
    print("PART 4  rendering the reference runs")
    print("=" * 72)
    OUT.mkdir(exist_ok=True)

    jobs = [
        ("7_3_plus_tracking_metrics.csv",
         ["reward_avg", "adv_avg", "adv_std", "entropy_avg"], "7_3_tracking.png"),
        ("7_4_plus_clip_ratio_metrics.csv",
         ["loss", "reward_avg", "avg_response_len", "eval_acc"], "7_4_clip.png"),
        ("7_5_plus_kl_metrics.csv",
         ["loss", "reward_avg", "entropy_avg", "eval_acc"], "7_5_kl.png"),
        ("7_6_plus_format_reward_metrics.csv",
         ["reward_avg", "format_reward_avg", "avg_response_len", "eval_acc"],
         "7_6_format.png"),
    ]
    for name, columns, out in jobs:
        if not (LOGS / name).exists():
            print(f"  skip {name} (missing)")
            continue
        plot_grpo_metrics(LOGS / name, columns, save_as=OUT / out)
        print(f"  {out:<20} {', '.join(columns)}")

    runs = sorted((Path(__file__).resolve().parent / "runs").glob("*/metrics.csv"))
    if runs:
        latest = runs[-1]
        plot_grpo_metrics(
            latest, ["loss", "reward_avg", "adv_std", "entropy_avg"],
            save_as=OUT / "my_latest_run.png")
        print(f"  {'my_latest_run.png':<20} from {latest.parent.name}")

    print(f"\n  written to {OUT}\n")


def main():
    part1_by_hand()
    part2_window_scales()
    part3_what_smoothing_hides()
    part4_render()


if __name__ == "__main__":
    main()
