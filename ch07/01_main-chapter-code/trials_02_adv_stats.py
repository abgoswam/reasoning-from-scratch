# Concept: advantage statistics (notebook cells 40-46, section 7.3.1)
#
# Chapter 6 computed advantages and used them. Chapter 7 starts *logging* two
# summary numbers per step -- adv_avg and adv_std -- and the whole point is
# that one of them is nearly useless and the other is the single best
# diagnostic in the run. This file shows which is which.
#
# Pure tensor math on hardcoded reward lists. No model, no GPU, instant.
#
#   Part 1: a healthy group
#   Part 2: the two degenerate groups
#   Part 3: why adv_avg is ~0 by construction
#   Part 4: what the 1e-4 floor is actually preventing
#   Part 5: reading it back off a real run

import torch


def compute_advantage_stats(rewards_list):
    """Identical to reasoning_from_scratch/ch07.py -- copied so this file stands alone."""
    rewards = torch.tensor(rewards_list)
    advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-4)
    return advantages, advantages.mean().item(), advantages.std().item()


def show(label, rewards):
    adv, avg, std = compute_advantage_stats(rewards)
    print(f"  {label:<22} rewards={rewards}")
    print(f"  {'':<22} advantages={[round(a, 3) for a in adv.tolist()]}")
    print(f"  {'':<22} adv_avg={avg:+.4f}  adv_std={std:.4f}")
    return adv, avg, std


# ---------------------------------------------------------------------------
# Part 1 -- a group where the rollouts disagree.
#
# Two of four rollouts solved the problem. Advantages come out symmetric:
# the winners get a positive push, the losers a negative one, and the size is
# roughly +/-1 because dividing by the std is what puts them on that scale.
# ---------------------------------------------------------------------------
def part1_healthy_group():
    print("=" * 72)
    print("PART 1  a group that carries signal")
    print("=" * 72)
    show("2 of 4 correct", [1., 1., 0., 0.])
    print("\n  -> this step produces a real gradient: two rollouts pushed up,")
    print("     two pushed down, relative to their own group\n")


# ---------------------------------------------------------------------------
# Part 2 -- the two ways a group carries NO signal.
#
# If every rollout scores the same, each reward equals the mean, so every
# numerator is 0 and every advantage is 0 -- regardless of whether they all
# failed or all succeeded. The step is a no-op: loss ~0, gradient ~0.
# ---------------------------------------------------------------------------
def part2_degenerate_groups():
    print("=" * 72)
    print("PART 2  the two degenerate groups")
    print("=" * 72)
    show("all failed", [0., 0., 0., 0.])
    print()
    show("all correct", [1., 1., 1., 1.])
    print("\n  -> identical advantages, opposite situations. GRPO cannot tell")
    print("     'too hard' from 'too easy' -- both waste the step.")
    print("     This is what --skip-zero-advantage-updates skips.\n")


# ---------------------------------------------------------------------------
# Part 3 -- adv_avg is ~0 whatever you feed it.
#
# Subtracting the mean is the FIRST thing the formula does, so the result is
# centered by construction. adv_avg is a sanity check that the arithmetic ran,
# not a signal about the model. adv_std is the informative one:
#
#     adv_std ~ 1.0  -> rollouts disagreed, the step taught the model something
#     adv_std ~ 0.0  -> rollouts agreed, the step was wasted
# ---------------------------------------------------------------------------
def part3_avg_is_always_zero():
    print("=" * 72)
    print("PART 3  adv_avg is ~0 by construction; adv_std is the real signal")
    print("=" * 72)

    cases = [
        ("2 of 4 correct", [1., 1., 0., 0.]),
        ("1 of 4 correct", [1., 0., 0., 0.]),
        ("3 of 4 correct", [1., 1., 1., 0.]),
        ("graded rewards", [1., 0.75, 0.25, 0.]),
        ("all failed", [0., 0., 0., 0.]),
        ("all correct", [1., 1., 1., 1.]),
    ]
    print(f"  {'group':<18} {'adv_avg':>10} {'adv_std':>10}   verdict")
    print("  " + "-" * 62)
    for label, rewards in cases:
        _, avg, std = compute_advantage_stats(rewards)
        verdict = "WASTED STEP" if std < 0.01 else "carries signal"
        print(f"  {label:<18} {avg:>+10.4f} {std:>10.4f}   {verdict}")

    print("\n  -> adv_avg never moves. But note adv_std does not move either,")
    print("     except between two values: ~0.9998 or ~0.0000.")
    print()
    print("     That is not a coincidence. Dividing by the std is what puts the")
    print("     advantages on unit scale, so their std comes back as ~1 for ANY")
    print("     group that disagreed at all -- whether the split was 1-of-4 or")
    print("     3-of-4, whether rewards were binary or graded.")
    print()
    print("     So adv_std is a BOOLEAN in disguise: 'did this group carry any")
    print("     signal?'. It does not measure how much. Do not read a trend")
    print("     into it -- read the fraction of steps where it is non-zero.\n")


# ---------------------------------------------------------------------------
# Part 4 -- the epsilon.
#
# When every reward is equal, std is 0 and the division is 0/0 -> NaN, which
# would poison the loss and every gradient after it. The 1e-4 turns that into
# 0/1e-4 = 0, so a degenerate group becomes a harmless no-op instead of
# destroying the run.
# ---------------------------------------------------------------------------
def part4_the_epsilon():
    print("=" * 72)
    print("PART 4  what the 1e-4 is preventing")
    print("=" * 72)

    rewards = torch.tensor([0., 0., 0., 0.])
    centered = rewards - rewards.mean()
    std = rewards.std()

    print(f"  rewards           : {rewards.tolist()}")
    print(f"  rewards - mean    : {centered.tolist()}   (all zero)")
    print(f"  rewards.std()     : {std.item()}          (also zero)")
    print()
    print(f"  without epsilon   : {(centered / std).tolist()}")
    print(f"  with    epsilon   : {(centered / (std + 1e-4)).tolist()}")
    print("\n  -> 0/0 is NaN. One NaN advantage makes the loss NaN, then every")
    print("     parameter NaN on the next optimizer step. The run is dead and")
    print("     the log looks fine right up until it isn't.\n")

    # It is genuinely 0/0, not a small number divided by a small number.
    nan_adv = centered / std
    print(f"  torch.isnan check : {torch.isnan(nan_adv).tolist()}\n")


# ---------------------------------------------------------------------------
# Part 5 -- what this looks like in a real metrics.csv.
#
# From a 10-step run of submit_03_tracking.py at num_rollouts=4: six of ten
# steps had adv_std=0. Those steps ran four full rollouts, a backward pass and
# an optimizer step, and changed the model by essentially nothing.
# ---------------------------------------------------------------------------
def part5_on_a_real_run():
    print("=" * 72)
    print("PART 5  reading adv_std off an actual run")
    print("=" * 72)

    observed = [  # step, reward_avg, adv_std  -- from runs/.../metrics.csv
        (1, 0.00, 0.000000), (2, 0.00, 0.000000), (3, 0.00, 0.000000),
        (4, 0.50, 0.999827), (5, 0.00, 0.000000), (6, 0.25, 0.999800),
        (7, 0.00, 0.000000), (8, 0.75, 0.999800), (9, 0.00, 0.000000),
        (10, 0.00, 0.000000),
    ]
    useful = 0
    print(f"  {'step':>4} {'reward_avg':>11} {'adv_std':>9}   contributed?")
    print("  " + "-" * 50)
    for step, reward, std in observed:
        ok = std > 0.01
        useful += ok
        print(f"  {step:>4} {reward:>11.2f} {std:>9.4f}   "
              f"{'yes' if ok else 'NO -- no-op'}")

    print(f"\n  {useful}/{len(observed)} steps produced a gradient "
          f"({100 * useful // len(observed)}%).")
    print("  More rollouts per step raises the odds that at least one differs,")
    print("  which is why the book recommends num_rollouts=8 and not 4.\n")


def main():
    part1_healthy_group()
    part2_degenerate_groups()
    part3_avg_is_always_zero()
    part4_the_epsilon()
    part5_on_a_real_run()


if __name__ == "__main__":
    main()
