# Concept: the format reward (notebook cells 148-149, section 7.6.1)
#
# reward_rlvr asks "is the answer right?". reward_format asks "did the model
# lay out its reasoning the way we asked?". Eleven lines, and most of the
# interesting behaviour is in the failure cases.
#
# Hardcoded token-id lists, so this needs no tokenizer and no model.
# trials_07_think_tokens.py covers where 151667/151668 come from.
#
#   Part 1: the function, read line by line
#   Part 2: every case it can hit
#   Part 3: what it does NOT check
#   Part 4: combining it with the correctness reward

import torch

THINK = 151667          # <think>
END_THINK = 151668      # </think>


def reward_format(token_ids, prompt_len, start_think_id=THINK,
                  end_think_id=END_THINK):
    """Identical to reasoning_from_scratch/ch07.py -- copied so this file stands alone."""
    try:
        gen = token_ids[prompt_len:].tolist()
        return float(gen.index(start_think_id) < gen.index(end_think_id))
    except ValueError:
        return 0.0


# ---------------------------------------------------------------------------
# Part 1 -- three ideas in one expression.
#
#   token_ids[prompt_len:]   look only at what the model GENERATED; a prompt
#                            containing <think> must not earn the reward
#   gen.index(x)             position of the FIRST occurrence
#   a < b                    the tags must appear in the right order
#
# and .index() raises ValueError when the tag is absent, which the except
# turns into 0.0. That try/except is the "tag missing" branch -- it is not
# defensive boilerplate.
# ---------------------------------------------------------------------------
def part1_line_by_line():
    print("=" * 74)
    print("PART 1  what each piece is doing")
    print("=" * 74)

    prompt = [100, 101, 102]
    gen = [200, THINK, 201, 202, END_THINK, 203]
    token_ids = torch.tensor(prompt + gen)

    print(f"  full sequence : {token_ids.tolist()}")
    print(f"  prompt_len    : {len(prompt)}")
    print(f"  generated     : {token_ids[len(prompt):].tolist()}")
    g = token_ids[len(prompt):].tolist()
    print(f"\n  gen.index({THINK})     -> {g.index(THINK)}")
    print(f"  gen.index({END_THINK})     -> {g.index(END_THINK)}")
    print(f"  {g.index(THINK)} < {g.index(END_THINK)}              -> "
          f"{g.index(THINK) < g.index(END_THINK)}")
    print(f"  float(...)             -> "
          f"{reward_format(token_ids, len(prompt))}\n")


# ---------------------------------------------------------------------------
# Part 2 -- the full truth table.
# ---------------------------------------------------------------------------
def part2_all_cases():
    print("=" * 74)
    print("PART 2  every case")
    print("=" * 74)

    prompt = [100, 101, 102]
    cases = [
        ("well formed", [200, THINK, 201, END_THINK, 202]),
        ("tags adjacent, empty", [THINK, END_THINK]),
        ("reversed order", [END_THINK, 201, THINK]),
        ("open tag only", [200, THINK, 201, 202]),
        ("close tag only", [200, 201, END_THINK]),
        ("no tags at all", [200, 201, 202]),
        ("empty generation", []),
        ("repeated, first pair ok", [THINK, 201, END_THINK, THINK, 202]),
        ("repeated, first pair bad", [END_THINK, THINK, 201, END_THINK]),
    ]
    print(f"  {'case':<26} {'reward':>7}   generated tokens")
    print("  " + "-" * 68)
    for label, gen in cases:
        r = reward_format(torch.tensor(prompt + gen, dtype=torch.long),
                          len(prompt))
        shown = [("<think>" if t == THINK else
                  "</think>" if t == END_THINK else t) for t in gen]
        print(f"  {label:<26} {r:>7.1f}   {shown}")

    print("\n  -> only FIRST occurrences count, so 'repeated, first pair bad'")
    print("     scores 0 even though a valid pair exists later in the text.\n")


# ---------------------------------------------------------------------------
# Part 3 -- what it deliberately does not check.
#
# This is a cheap shaping signal, not a validator. It says nothing about
# whether the tags are balanced, whether anything sits between them, or
# whether the answer that follows is correct.
# ---------------------------------------------------------------------------
def part3_what_it_ignores():
    print("=" * 74)
    print("PART 3  what it does not check")
    print("=" * 74)

    prompt = [100]
    degenerate = [
        ("nothing between the tags", [THINK, END_THINK]),
        ("three opens, one close", [THINK, THINK, THINK, END_THINK]),
        ("tags then nothing else", [THINK, END_THINK]),
        ("no answer after the tags", [THINK, 5, END_THINK]),
    ]
    for label, gen in degenerate:
        r = reward_format(torch.tensor(prompt + gen, dtype=torch.long), len(prompt))
        print(f"  {r:>5.1f}   {label}")

    print("\n  -> a model can farm this reward by emitting '<think></think>'")
    print("     and stopping. That is exactly why it is ADDED to the")
    print("     correctness reward rather than replacing it: format alone")
    print("     scores 1.0, but format plus a wrong answer still loses to")
    print("     format plus a right one.\n")


# ---------------------------------------------------------------------------
# Part 4 -- how it enters the loss.
#
#     reward = rlvr_reward + format_reward_weight * format_reward
#
# and then the usual group-relative advantage runs on that sum. So the weight
# controls how much of the learning signal is spent on layout instead of
# correctness.
# ---------------------------------------------------------------------------
def part4_combining():
    print("=" * 74)
    print("PART 4  reward = rlvr + weight * format")
    print("=" * 74)

    rollouts = [("correct + tagged", 1.0, 1.0), ("correct, no tags", 1.0, 0.0),
                ("wrong + tagged", 0.0, 1.0), ("wrong, no tags", 0.0, 0.0)]

    for weight in (0.0, 0.5, 1.0):
        rewards = torch.tensor([r + weight * f for _, r, f in rollouts])
        adv = (rewards - rewards.mean()) / (rewards.std() + 1e-4)
        print(f"\n  format_reward_weight = {weight}")
        print(f"    {'rollout':<20} {'rlvr':>5} {'fmt':>5} "
              f"{'total':>7} {'advantage':>10}")
        for (label, r, f), tot, a in zip(rollouts, rewards, adv):
            print(f"    {label:<20} {r:>5.1f} {f:>5.1f} "
                  f"{tot:>7.2f} {a:>+10.3f}")

    print("\n  -> at weight=1.0 'wrong + tagged' and 'correct, no tags' tie at")
    print("     1.0, so the group cannot tell them apart. Layout is being")
    print("     valued exactly as much as being right. Lower the weight once")
    print("     the format has been learned.\n")


def main():
    part1_line_by_line()
    part2_all_cases()
    part3_what_it_ignores()
    part4_combining()


if __name__ == "__main__":
    main()
