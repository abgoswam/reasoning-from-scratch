# Concept: entropy tracking (notebook cells 53-64, section 7.3.2)
#
# Entropy answers "how undecided is the model at this step?". It is the single
# best early warning for policy collapse: a model that stops exploring will
# show entropy sliding toward zero long before the reward curve reacts.
#
# Seven hardcoded logits. No model, no GPU, instant.
#
#   Part 1: logits -> logprobs -> probs, and picking a token
#   Part 2: entropy as -sum(p * log p)
#   Part 3: the two extremes that bound it
#   Part 4: temperature, and what collapse looks like
#   Part 5: per-step entropy vs. the one number the CSV logs

import torch

# The chapter's toy distribution over a 7-token vocabulary.
LOGITS = torch.tensor(
    [0.6667, -2.0000, 1.3333, -0.0000, -0.6667, 2.0000, -1.3333]
)


# ---------------------------------------------------------------------------
# Part 1 -- the same three views of one distribution.
#
# logits are unbounded scores; softmax turns them into probabilities that sum
# to 1; log_softmax gives the logs of those same probabilities directly and
# more stably than log(softmax(x)).
# ---------------------------------------------------------------------------
def part1_three_views():
    print("=" * 72)
    print("PART 1  logits, logprobs, probs")
    print("=" * 72)

    logprobs = torch.log_softmax(LOGITS, dim=-1)
    probs = torch.softmax(LOGITS, dim=-1)

    print(f"  {'id':>3} {'logit':>9} {'logprob':>10} {'prob':>9}")
    print("  " + "-" * 34)
    for i in range(len(LOGITS)):
        print(f"  {i:>3} {LOGITS[i]:>9.4f} {logprobs[i]:>10.4f} {probs[i]:>9.4f}")

    print(f"\n  probs sum to        : {probs.sum().item():.6f}")
    print(f"  exp(logprobs) == probs: {torch.allclose(logprobs.exp(), probs)}")

    selected = torch.argmax(logprobs)
    print(f"\n  most likely token id : {selected.item()}")
    print(f"  its logprob          : {logprobs[selected].item():.4f}")
    print(f"  its probability      : {probs[selected].item():.4f}")
    print("\n  -> ch06 only ever needed the logprob of the token that WAS")
    print("     chosen. Entropy needs the whole row.\n")


# ---------------------------------------------------------------------------
# Part 2 -- entropy.
#
#     H = -sum_v  p_v * log p_v
#
# Each term is a probability weighted by its own surprise. A token the model
# is sure about contributes almost nothing (p*log p -> 0 as p -> 1); a token
# it is unsure about contributes a lot.
# ---------------------------------------------------------------------------
def part2_entropy():
    print("=" * 72)
    print("PART 2  entropy = -sum(p * log p)")
    print("=" * 72)

    probs = torch.softmax(LOGITS, dim=-1)
    logprobs = torch.log_softmax(LOGITS, dim=-1)
    terms = -(probs * logprobs)

    print(f"  {'id':>3} {'prob':>9} {'-p*log p':>11}")
    print("  " + "-" * 25)
    for i in range(len(LOGITS)):
        print(f"  {i:>3} {probs[i]:>9.4f} {terms[i]:>11.4f}")

    entropy = terms.sum()
    print(f"\n  entropy = {entropy.item():.4f} nats")
    print(f"  matches torch.sum(-(probs * logprobs)): "
          f"{torch.allclose(entropy, torch.sum(-(probs * logprobs)))}")
    # -p*log(p) is 0 at both p=0 and p=1, and peaks in between at p = 1/e.
    peak = 1.0 / torch.e
    print(f"\n  the term -p*log(p) peaks at p = 1/e = {peak:.4f}, and is 0 at")
    print("  both p=0 and p=1. So a token contributes nothing when the model")
    print("  is certain about it AND nothing when it is irrelevant --")
    print("  entropy is built almost entirely from the mid-range tokens.")
    print(f"  Here ids 5 (p={probs[5]:.3f}) and 2 (p={probs[2]:.3f}) straddle")
    print(f"  that peak and together supply "
          f"{100 * (terms[5] + terms[2]) / entropy:.0f}% of the total.\n")


# ---------------------------------------------------------------------------
# Part 3 -- the bounds, so the number means something.
#
#   uniform over V tokens -> H = log(V)   (maximum: no idea at all)
#   one-hot               -> H = 0        (minimum: fully decided)
# ---------------------------------------------------------------------------
def part3_bounds():
    print("=" * 72)
    print("PART 3  what counts as high or low")
    print("=" * 72)

    def H(logits):
        p = torch.softmax(logits, dim=-1)
        return torch.sum(-(p * torch.log_softmax(logits, dim=-1))).item()

    V = len(LOGITS)
    uniform = torch.zeros(V)
    onehot = torch.tensor([0.] * (V - 1) + [50.])

    print(f"  uniform over {V} tokens : H = {H(uniform):.4f}   "
          f"(= log({V}) = {torch.log(torch.tensor(float(V))).item():.4f})")
    print(f"  our toy distribution   : H = {H(LOGITS):.4f}")
    print(f"  near one-hot           : H = {H(onehot):.4f}")

    print(f"\n  For Qwen3's real vocabulary of 151,936 the ceiling is "
          f"log(151936) = {torch.log(torch.tensor(151936.)).item():.2f}.")
    print("  Real runs sit far below that, because most of the vocabulary is")
    print("  irrelevant at any given position. Judge entropy by its TREND,")
    print("  not against the theoretical maximum.\n")


# ---------------------------------------------------------------------------
# Part 4 -- temperature is the dial, entropy is the readout.
#
# Sampling divides logits by temperature before the softmax. Watching entropy
# fall as temperature falls is the same shape as a policy collapsing: the
# distribution concentrates and the model stops exploring.
# ---------------------------------------------------------------------------
def part4_temperature():
    print("=" * 72)
    print("PART 4  temperature, and the shape of collapse")
    print("=" * 72)

    print(f"  {'temperature':>12} {'entropy':>9} {'top prob':>10}")
    print("  " + "-" * 34)
    for t in (2.0, 1.0, 0.8, 0.5, 0.2, 0.05):
        scaled = LOGITS / t
        p = torch.softmax(scaled, dim=-1)
        h = torch.sum(-(p * torch.log_softmax(scaled, dim=-1))).item()
        print(f"  {t:>12.2f} {h:>9.4f} {p.max().item():>10.4f}")

    print("\n  -> training at temperature 0.8 fixes the sampling distribution,")
    print("     but the WEIGHTS can drift the same way. A run whose entropy")
    print("     slides steadily toward 0 has stopped exploring: it emits the")
    print("     same tokens every rollout, every group agrees, adv_std goes")
    print("     to 0, and learning stops. See trials_02 part 3.\n")


# ---------------------------------------------------------------------------
# Part 5 -- from one position to one number per step.
#
# ch07.py's sequence_logprob_and_entropy computes entropy at EVERY generated
# position, then averages. So entropy_avg in the CSV is:
#
#     mean over answer tokens of ( -sum over vocab of p*log p )
#
# Two averages, in that order. Averaging hides the shape: a rollout that is
# confident for 500 tokens then wildly unsure for 12 reports the same number
# as one that is mildly unsure throughout.
# ---------------------------------------------------------------------------
def part5_per_step_then_mean():
    print("=" * 72)
    print("PART 5  per-position entropy -> one number per step")
    print("=" * 72)

    torch.manual_seed(0)
    # Five generated positions over our 7-token vocabulary.
    per_position = torch.stack([LOGITS * s for s in (0.2, 0.5, 1.0, 2.0, 4.0)])
    logprobs = torch.log_softmax(per_position, dim=-1)
    probs = torch.exp(logprobs)
    step_entropy = -torch.sum(probs * logprobs, dim=-1)   # sum over vocab

    print(f"  logits shape        : {tuple(per_position.shape)}  "
          f"(positions, vocab)")
    print(f"  per-position entropy: "
          f"{[round(v, 3) for v in step_entropy.tolist()]}")
    print(f"  entropy_avg (logged): {step_entropy.mean().item():.4f}")

    print("\n  -> the CSV column is that last number. In the 10-step run it")
    print("     went 0.755, 1.079, 1.426, 0.707, 0.606, 0.111, 0.181, ...")
    print("     Step 6 at 0.111 was the model going nearly deterministic")
    print("     while generating 500 tokens. Over 10 steps that is noise;")
    print("     sustained, it is the failure that 7.4 and 7.5 exist to stop.\n")


def main():
    part1_three_views()
    part2_entropy()
    part3_bounds()
    part4_temperature()
    part5_per_step_then_mean()


if __name__ == "__main__":
    main()
