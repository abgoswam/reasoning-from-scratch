# Concept: reading a GRPO training run (notebook cells 13-31, section 7.2)
#
# Chapter 7 opens by NOT training. It hands you finished runs and teaches you
# to read them. That is the actual skill: a GRPO run fails quietly, and the
# failure is visible in the metrics long before the reward curve reacts.
#
# Reads the five reference logs shipped in ch07/02_logs/. Nothing is
# downloaded -- the notebook's download_from_github calls exist for readers
# who did not clone the repo.
#
#   Part 1: the CSV headers ARE the chapter outline
#   Part 2: the ch06 baseline, summarised
#   Part 3: the four columns that matter, and what "healthy" looks like
#   Part 4: eval_acc is sparse on purpose
#   Part 5: the five runs side by side -- including one that collapses
#   Part 6: compare your own run against the reference

import csv
from pathlib import Path

LOGS = Path(__file__).resolve().parent.parent / "02_logs"

STAGES = [
    ("ch06 baseline", "ch06_rlvr_grpo_original_no_kl_metrics.csv"),
    ("7.3 tracking", "7_3_plus_tracking_metrics.csv"),
    ("7.4 clip ratio", "7_4_plus_clip_ratio_metrics.csv"),
    ("7.5 + KL", "7_5_plus_kl_metrics.csv"),
    ("7.6 + format", "7_6_plus_format_reward_metrics.csv"),
]


def read_csv(path):
    with Path(path).open(newline="") as f:
        return [r for r in csv.DictReader(f) if r.get("step")]


def col(rows, name):
    return [float(r[name]) for r in rows if r.get(name)]


# ---------------------------------------------------------------------------
# Part 1 -- each stage adds exactly the columns its new mechanism produces.
#
# You can read the whole chapter's arc off the headers without opening a
# single plot.
# ---------------------------------------------------------------------------
def part1_headers_are_the_outline():
    print("=" * 76)
    print("PART 1  the CSV headers are the chapter outline")
    print("=" * 76)

    seen = []
    for label, name in STAGES:
        path = LOGS / name
        if not path.exists():
            print(f"  {label:<16} MISSING: {path}")
            continue
        header = path.read_text().splitlines()[0].split(",")
        added = [c for c in header if c not in seen]
        seen = header
        print(f"  {label:<16} +{', '.join(added) if added else '(nothing new)'}")

    print("\n  -> adv_avg/adv_std/entropy_avg arrive with 7.3, policy_ratio")
    print("     with 7.4, kl_loss with 7.5, format_reward_avg with 7.6.")
    print("     Every stage is a strict superset of the one before it.\n")


# ---------------------------------------------------------------------------
# Part 2 -- the run chapter 7 is reacting to.
# ---------------------------------------------------------------------------
def part2_baseline_summary():
    print("=" * 76)
    print("PART 2  the chapter 6 baseline run")
    print("=" * 76)

    rows = read_csv(LOGS / STAGES[0][1])
    print(f"  steps logged : {len(rows)}")
    print()
    print(f"  {'column':<20} {'first':>10} {'last':>10} {'min':>10} {'max':>10}")
    print("  " + "-" * 64)
    for name in ("loss", "reward_avg", "avg_response_len", "tokens_per_sec"):
        v = col(rows, name)
        if v:
            print(f"  {name:<20} {v[0]:>10.3f} {v[-1]:>10.3f} "
                  f"{min(v):>10.3f} {max(v):>10.3f}")
    print()


# ---------------------------------------------------------------------------
# Part 3 -- what to actually look at.
#
# reward_avg is the outcome, but it is noisy and lags. The other three tell
# you WHY it is moving, and they move first.
# ---------------------------------------------------------------------------
def part3_what_healthy_looks_like():
    print("=" * 76)
    print("PART 3  the four columns, and what each one is for")
    print("=" * 76)

    guide = [
        ("reward_avg", "the outcome. Noisy; judge it over a window, not per step."),
        ("loss", "NOT a quality measure. A GRPO loss near 0 usually means the"),
        ("", "  group was degenerate, not that the model is good."),
        ("avg_response_len", "collapsing toward ~5 means the model is emitting"),
        ("", "  near-empty answers; pinned at max_new_tokens means it never stops."),
        ("entropy_avg", "sliding toward 0 is policy collapse. Earliest warning."),
    ]
    for name, text in guide:
        print(f"  {name:<18} {text}")

    print()
    rows = read_csv(LOGS / STAGES[1][1])          # 7.3 has the extra columns
    n = len(rows)
    for name in ("reward_avg", "entropy_avg", "avg_response_len"):
        v = col(rows, name)
        if not v:
            continue
        q = max(1, n // 4)
        quarters = [sum(v[i:i + q]) / len(v[i:i + q]) for i in range(0, len(v), q)][:4]
        print(f"  {name:<18} by quarter: "
              + "  ".join(f"{x:>8.3f}" for x in quarters))

    adv_std = col(rows, "adv_std")
    if adv_std:
        wasted = sum(1 for x in adv_std if x < 0.01)
        print(f"\n  adv_std == 0 on {wasted}/{len(adv_std)} steps "
              f"({100 * wasted // len(adv_std)}% of updates did nothing)")
    print()


# ---------------------------------------------------------------------------
# Part 4 -- eval_acc has holes, and that is deliberate.
#
# Running MATH-500 costs a full generation pass per example, so it happens
# only at checkpoints. plot_grpo_metrics draws it as a bar chart for exactly
# this reason -- a line would imply values that were never measured.
# ---------------------------------------------------------------------------
def part4_sparse_eval():
    print("=" * 76)
    print("PART 4  eval_acc is sparse")
    print("=" * 76)

    for label, name in STAGES:
        path = LOGS / name
        if not path.exists():
            continue
        rows = read_csv(path)
        present = [(int(r["step"]), float(r["eval_acc"]))
                   for r in rows if r.get("eval_acc")]
        if not present:
            print(f"  {label:<16} no eval_acc values")
            continue
        steps = ", ".join(str(s) for s, _ in present[:6])
        print(f"  {label:<16} {len(present):>3} of {len(rows):>4} steps  "
              f"at steps {steps}{' ...' if len(present) > 6 else ''}")
        print(f"  {'':<16} MATH-500 accuracy {present[0][1]:.1f}% "
              f"-> {present[-1][1]:.1f}%")
    print()


# ---------------------------------------------------------------------------
# Part 5 -- the payoff of having all five logs: read them together.
#
# This is where chapter 7 makes its case, and where one run fails badly enough
# to be worth studying.
# ---------------------------------------------------------------------------
def part5_side_by_side():
    print("=" * 76)
    print("PART 5  five runs, side by side (MATH-500 accuracy %)")
    print("=" * 76)

    series = {}
    for label, name in STAGES:
        path = LOGS / name
        if not path.exists():
            continue
        rows = read_csv(path)
        series[label] = [float(r["eval_acc"]) for r in rows if r.get("eval_acc")]
        print(f"  {label:<16} " + " ".join(f"{a:>5.1f}" for a in series[label]))

    print()
    base = series.get("ch06 baseline")
    track = series.get("7.3 tracking")
    if base and track:
        print(f"  ch06 and 7.3 are identical: {base == track}")
        print("    -> 7.3 only ADDS metrics. Tracking changes nothing about")
        print("       training, which is exactly what you want from tracking.\n")

    clip = series.get("7.4 clip ratio")
    if base and clip:
        print(f"  ch06 peaks at {max(base):.1f}% then decays to {base[-1]:.1f}%")
        print(f"  7.4  peaks at {max(clip):.1f}% and holds  {clip[-1]:.1f}%")
        print("    -> clipping did not raise the ceiling; it stopped the decay.")
        print("       That is what a stabiliser is supposed to look like.\n")

    kl = series.get("7.5 + KL")
    if kl:
        dead = sum(1 for a in kl if a == 0.0)
        print(f"  7.5 + KL: {kl[0]:.1f}% -> {max(kl):.1f}% -> 0.0% for the "
              f"last {dead} evaluations")
        print("    -> this run COLLAPSED. Not a plotting artifact: at step 500")
        print("       the log reads")
        print("           entropy_avg=11.92  avg_response_len=1024.0  reward_avg=0.000")
        print()
        print("       11.92 is the number trials_03 part 3 computes as the")
        print("       ceiling: log(151936) = 11.93. The policy went to a")
        print("       near-UNIFORM distribution over the whole vocabulary and")
        print("       emitted max-length noise on every rollout.")
        print()
        print("       Note the direction. Collapse is usually described as")
        print("       entropy going to 0 (the model repeats itself). Here it")
        print("       ran the other way, to the maximum. Both are collapse:")
        print("       the policy stopped being a policy.")
        print()
        print("       A KL penalty is supposed to PREVENT drift, so this is a")
        print("       case of the stabiliser making things worse -- which is")
        print("       why 7.5's default is kl_coeff=0.0 on the command line")
        print("       even though the function signature says 0.02.\n")

    fmt = series.get("7.6 + format")
    if fmt:
        print(f"  7.6 starts at {fmt[0]:.1f}%, far above the others")
        print("    -> because it trains the REASONING model, not the base one.")
        print("       Do not read 7.5 -> 7.6 as an improvement; it is a")
        print("       different starting point.\n")


# ---------------------------------------------------------------------------
# Part 6 -- your own runs.
#
# submit_03_tracking.py writes the same columns to
# runs/<timestamp>-<wandb-id>/metrics.csv, so everything above applies
# unchanged to what you produce.
# ---------------------------------------------------------------------------
def part6_your_runs():
    print("=" * 76)
    print("PART 6  your own runs")
    print("=" * 76)

    runs = sorted((Path(__file__).resolve().parent / "runs").glob("*/metrics.csv"))
    if not runs:
        print("  No local runs yet. Produce one with:")
        print("    python submit_03_tracking.py --steps 10 --num_rollouts 4 \\")
        print("        --device cuda:1 --wandb_project rfs-ch07\n")
        return

    print(f"  {'run':<34} {'steps':>6} {'reward':>8} {'wasted':>8}")
    print("  " + "-" * 60)
    for path in runs:
        rows = read_csv(path)
        if not rows:
            continue
        reward = col(rows, "reward_avg")
        std = col(rows, "adv_std")
        wasted = (f"{100 * sum(1 for x in std if x < 0.01) // len(std)}%"
                  if std else "n/a")
        print(f"  {path.parent.name:<34} {len(rows):>6} "
              f"{sum(reward) / len(reward):>8.3f} {wasted:>8}")

    print("\n  -> compare against the reference runs above. A short run will")
    print("     look terrible; that is expected, not a bug.\n")


def main():
    part1_headers_are_the_outline()
    part2_baseline_summary()
    part3_what_healthy_looks_like()
    part4_sparse_eval()
    part5_side_by_side()
    part6_your_runs()


if __name__ == "__main__":
    main()
