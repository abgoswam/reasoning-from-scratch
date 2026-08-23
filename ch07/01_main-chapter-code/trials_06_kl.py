# Concept: the KL loss term (notebook cell 112, section 7.5.1)
#
# A KL penalty is supposed to keep the policy from wandering too far from
# where it started. It costs a second frozen copy of the model and a third
# forward pass per rollout -- and in the chapter's own reference run it made
# things dramatically worse. Worth understanding before you switch it on.
#
# Toy tensors and a 12-parameter toy model. No real model, no GPU, instant.
#
#   Part 1: the estimator, which is just a mean of log-ratios
#   Part 2: it can go negative -- true KL cannot
#   Part 3: building the reference model, and proving it is frozen
#   Part 4: what kl_coeff does to the loss
#   Part 5: what it costs, and the run where it backfired

import copy

import torch

NEW_LOGPS = torch.tensor([-7.9243, -20.1546, -16.6130, -23.3677])
REF_LOGPS = torch.tensor([-8.4021, -19.8770, -17.2044, -22.9013])


# ---------------------------------------------------------------------------
# Part 1 -- the whole KL term is one line in 7_5_plus_kl.py:
#
#     kl_loss = kl_coeff * torch.mean(new_logps - ref_logps)
#
# Because logp is a log, the subtraction is a log-RATIO: how much more (or
# less) likely the current policy makes this rollout than the frozen
# reference did. Averaged over rollouts, that is the "k1" estimator of
# KL(new || ref).
# ---------------------------------------------------------------------------
def part1_the_estimator():
    print("=" * 74)
    print("PART 1  kl = mean(new_logp - ref_logp)")
    print("=" * 74)

    diff = NEW_LOGPS - REF_LOGPS
    ratio = torch.exp(diff)

    print(f"  {'i':>3} {'ref_logp':>10} {'new_logp':>10} {'diff':>8} {'ratio':>8}")
    print("  " + "-" * 44)
    for i in range(len(diff)):
        print(f"  {i:>3} {REF_LOGPS[i]:>10.4f} {NEW_LOGPS[i]:>10.4f} "
              f"{diff[i]:>+8.4f} {ratio[i]:>8.4f}")

    print(f"\n  kl estimate = mean(diff) = {diff.mean().item():+.4f}")
    print("\n  -> positive means the policy has drifted toward these rollouts;")
    print("     the penalty pushes back. Note it is the same subtraction as")
    print("     the policy ratio in trials_05 -- different reference point.\n")


# ---------------------------------------------------------------------------
# Part 2 -- the estimator is unbiased but noisy, and a finite sample of it
# can come out NEGATIVE. True KL divergence is >= 0 by definition, so a
# negative kl_loss in the CSV is not a bug -- it is a small sample.
# ---------------------------------------------------------------------------
def part2_can_go_negative():
    print("=" * 74)
    print("PART 2  a negative kl_loss is not a bug")
    print("=" * 74)

    cases = [
        ("drifted toward", torch.tensor([-7.0, -19.0, -16.0, -22.0])),
        ("drifted away", torch.tensor([-9.5, -21.0, -18.5, -24.0])),
        ("barely moved", REF_LOGPS + torch.tensor([0.01, -0.02, 0.01, -0.01])),
    ]
    for label, new in cases:
        est = (new - REF_LOGPS).mean().item()
        print(f"  {label:<16} kl estimate = {est:+.4f}"
              f"{'   <- NEGATIVE' if est < 0 else ''}")

    print("\n  -> real KL(p||q) = sum p*log(p/q) >= 0 always. This estimator")
    print("     averages log-ratios over the few rollouts you happened to")
    print("     sample, so it scatters around the true value and can land")
    print("     below zero. With num_rollouts=4 that scatter is large.\n")


# ---------------------------------------------------------------------------
# Part 3 -- the reference model.
#
#     ref_model = copy.deepcopy(model).to(device)
#     ref_model.eval()
#     for p in ref_model.parameters():
#         p.requires_grad = False
#
# Three separate things: an independent copy, eval mode, and no gradients.
# Skip the third and backward would try to push gradients into the reference.
# ---------------------------------------------------------------------------
class ToyPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.head = torch.nn.Linear(3, 4, bias=False)

    def forward(self, x):
        return torch.log_softmax(self.head(x), dim=-1)


def part3_reference_model():
    print("=" * 74)
    print("PART 3  the frozen reference copy")
    print("=" * 74)

    torch.manual_seed(0)
    model = ToyPolicy()

    ref_model = copy.deepcopy(model)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False

    print(f"  same object?              {model is ref_model}")
    print(f"  same weight tensor?       "
          f"{model.head.weight is ref_model.head.weight}")
    print(f"  same values right now?    "
          f"{torch.equal(model.head.weight, ref_model.head.weight)}")
    print(f"  model requires_grad       {model.head.weight.requires_grad}")
    print(f"  ref   requires_grad       {ref_model.head.weight.requires_grad}")

    x = torch.randn(3)
    new_logp = model(x).sum()
    with torch.no_grad():
        ref_logp = ref_model(x).sum()
    loss = -(new_logp) + 0.02 * (new_logp - ref_logp)
    loss.backward()

    print(f"\n  after backward:")
    print(f"    model.head.weight.grad  {'set' if model.head.weight.grad is not None else 'None'}")
    print(f"    ref_model...grad        {'set' if ref_model.head.weight.grad is not None else 'None'}")

    with torch.no_grad():
        model.head.weight += 1.0
    print(f"\n  after changing the model's weights:")
    print(f"    still equal?            "
          f"{torch.equal(model.head.weight, ref_model.head.weight)}")
    print("\n  -> deepcopy is what makes them independent. A plain assignment")
    print("     would alias the same tensors and the penalty would compare")
    print("     the model against itself: kl == 0 forever.\n")


# ---------------------------------------------------------------------------
# Part 4 -- kl_coeff is the dial between "follow the reward" and "stay put".
# ---------------------------------------------------------------------------
def part4_coeff_sweep():
    print("=" * 74)
    print("PART 4  kl_coeff")
    print("=" * 74)

    pg_loss = -2.5764                       # from trials_05 part 1
    kl = (NEW_LOGPS - REF_LOGPS).mean().item()

    print(f"  pg_loss     = {pg_loss:+.4f}")
    print(f"  kl estimate = {kl:+.4f}\n")
    print(f"  {'kl_coeff':>10} {'kl_loss':>10} {'total':>10}  share of loss")
    print("  " + "-" * 50)
    for c in (0.0, 0.001, 0.02, 0.1, 1.0):
        kl_loss = c * kl
        total = pg_loss + kl_loss
        share = abs(kl_loss) / (abs(pg_loss) + abs(kl_loss) + 1e-12)
        print(f"  {c:>10.3f} {kl_loss:>10.4f} {total:>10.4f}  {100*share:>5.1f}%")

    print("\n  -> 7_5_plus_kl.py uses kl_coeff=0.001 everywhere -- function")
    print("     signature AND command line -- so its default run DOES apply")
    print("     the penalty. 7_6 changes the signature to 0.02 but the CLI")
    print("     default to 0.0, so check both before assuming.\n")


# ---------------------------------------------------------------------------
# Part 5 -- the price, and the cautionary tale.
# ---------------------------------------------------------------------------
def part5_cost_and_caution():
    print("=" * 74)
    print("PART 5  what it costs, and the run where it backfired")
    print("=" * 74)

    params = 751_632_384                    # measured, not quoted
    bf16 = params * 2 / 1024 ** 3
    print(f"  Qwen3-0.6B parameters      : {params:,}")
    print(f"  a second frozen copy (bf16): {bf16:.2f} GiB resident for the")
    print(f"  whole run, plus one extra forward pass per rollout.")
    print(f"  (\"0.6B\" is the marketing name; the tied-embedding model")
    print(f"   actually holds {params / 1e9:.2f}B parameters.)\n")

    print("  And from ch07/02_logs/7_5_plus_kl_metrics.csv (see trials_01):")
    print("    MATH-500 accuracy 15.6% -> 40.8% -> 0.0% for the rest of the run")
    print("    at step 500: entropy_avg=11.92  avg_response_len=1024  reward=0.000")
    print()
    print("  11.92 is log(151936), the entropy ceiling from trials_03 part 3:")
    print("  the policy went uniform over the entire vocabulary and emitted")
    print("  max-length noise.")
    print()
    print("  -> a KL penalty is a stabiliser that can destabilise. DAPO and")
    print("     Dr. GRPO both recommend dropping it entirely for math (items")
    print("     4 and 8 in the 7.7 list). Switch it on deliberately, watch")
    print("     entropy_avg, and be ready to turn it back off.\n")


def main():
    part1_the_estimator()
    part2_can_go_negative()
    part3_reference_model()
    part4_coeff_sweep()
    part5_cost_and_caution()


if __name__ == "__main__":
    main()
