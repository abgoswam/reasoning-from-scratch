# Concept: sequence_logprob, the shift-by-one, and the prompt mask
#
# Companion to trials_06_logprobs.py, which runs the real thing on the real
# model. This one uses a 6-word vocabulary and a 5-parameter toy model so
# every tensor fits on screen and every number can be checked by hand.
#
# Put a breakpoint anywhere and inspect -- nothing here is bigger than [6, 6].
#
#   Part 1: why position i scores token i+1  (the shift)
#   Part 2: three ways to select those logprobs, all identical
#   Part 3: why the slice starts at prompt_len - 1, not prompt_len
#   Part 4: sum vs. mean, and why GRPO wants the sum
#
# The real implementation being explained (trials_00.py:117):
#
#     def sequence_logprob(model, token_ids, prompt_len):
#         logits = model(token_ids.unsqueeze(0)).squeeze(0).float()
#         logprobs = torch.log_softmax(logits, dim=-1)
#         selected = logprobs[:-1].gather(1, token_ids[1:].unsqueeze(-1)).squeeze(-1)
#         return torch.sum(selected[prompt_len - 1:])

import torch

from trials_00 import sequence_logprob, sequence_logprob_draft

VOCAB = ["the", "cat", "sat", "on", "a", "mat"]
V = len(VOCAB)

# "the cat" is the prompt, "sat on a mat" is the completion.
TOKEN_IDS = torch.tensor([0, 1, 2, 3, 4, 5])
PROMPT_LEN = 2


class ToyModel(torch.nn.Module):
    """Same interface as Qwen3Model: [batch, seq] ids -> [batch, seq, vocab]."""

    def __init__(self, vocab_size, emb_dim=4):
        super().__init__()
        self.emb = torch.nn.Embedding(vocab_size, emb_dim)
        self.out_head = torch.nn.Linear(emb_dim, vocab_size, bias=False)

    def forward(self, in_idx):
        return self.out_head(self.emb(in_idx))


def show(name, tensor):
    print(f"{name:<26} shape={tuple(tensor.shape)}")


# ---------------------------------------------------------------------------
# Part 1 -- the shift.
#
# A language model is trained so that the output at position i predicts the
# token at position i+1. So the logits row for "the" (position 0) holds the
# distribution over what comes AFTER "the" -- and the token that actually came
# after is "cat" at position 1.
#
# That off-by-one is the entire reason for the two slices in sequence_logprob:
#     logprobs[:-1]   drop the last row  -- it predicts a token we don't have
#     token_ids[1:]   drop the first id  -- nothing predicts the first token
# ---------------------------------------------------------------------------
def part1_the_shift(model):
    print("=" * 72)
    print("PART 1  position i predicts token i+1")
    print("=" * 72)

    logits = model(TOKEN_IDS.unsqueeze(0)).squeeze(0).float()
    logprobs = torch.log_softmax(logits, dim=-1)
    show("logits", logits)
    show("logprobs", logprobs)
    print()

    print(f"  {'pos':>3}  {'token':<6} {'predicts pos':>12}  "
          f"{'target':<6} {'logprob':>9}")
    for i in range(len(TOKEN_IDS)):
        token = VOCAB[TOKEN_IDS[i]]
        if i == len(TOKEN_IDS) - 1:
            print(f"  {i:>3}  {token:<6} {i + 1:>12}  "
                  f"{'--':<6} {'DROPPED':>9}   <- no token at pos 6")
            continue
        target_id = TOKEN_IDS[i + 1]
        value = logprobs[i, target_id].item()
        print(f"  {i:>3}  {token:<6} {i + 1:>12}  "
              f"{VOCAB[target_id]:<6} {value:>9.4f}")

    print(f"\n  first token '{VOCAB[TOKEN_IDS[0]]}' is never scored -- "
          f"nothing precedes it")
    print(f"  so {len(TOKEN_IDS)} tokens yield {len(TOKEN_IDS) - 1} logprobs\n")


# ---------------------------------------------------------------------------
# Part 2 -- three spellings of "pick logprobs[i, token_ids[i+1]] for all i".
#
# gather(1, index) means: out[i, j] = logprobs[i, index[i, j]].
# With index of shape [T-1, 1] we pull exactly one entry per row, then
# squeeze(-1) turns [T-1, 1] back into [T-1].
# ---------------------------------------------------------------------------
def part2_three_spellings(model):
    print("=" * 72)
    print("PART 2  loop vs. arange vs. gather")
    print("=" * 72)

    logits = model(TOKEN_IDS.unsqueeze(0)).squeeze(0).float()
    logprobs = torch.log_softmax(logits, dim=-1)

    by_loop = torch.stack([
        logprobs[i, TOKEN_IDS[i + 1]] for i in range(len(TOKEN_IDS) - 1)
    ])

    positions = torch.arange(len(TOKEN_IDS) - 1)
    by_arange = logprobs[positions, TOKEN_IDS[1:]]

    index = TOKEN_IDS[1:].unsqueeze(-1)
    show("logprobs[:-1]", logprobs[:-1])
    show("index = token_ids[1:]...", index)
    by_gather = logprobs[:-1].gather(1, index).squeeze(-1)
    show("after gather + squeeze", by_gather)
    print()

    print("  loop  :", [f"{v:.4f}" for v in by_loop.tolist()])
    print("  arange:", [f"{v:.4f}" for v in by_arange.tolist()])
    print("  gather:", [f"{v:.4f}" for v in by_gather.tolist()])
    print(f"\n  all identical: "
          f"{torch.allclose(by_loop, by_arange) and torch.allclose(by_arange, by_gather)}")
    print("  gather is used because it is one fused CUDA kernel, not T lookups\n")


# ---------------------------------------------------------------------------
# Part 3 -- the prompt mask, and the -1 that trips everyone up.
#
# We only want gradient on tokens the POLICY chose. The prompt was given, not
# generated, so its logprobs must not enter the loss.
#
# After the shift, `selected[j]` means "logits at position j scored the token
# at position j+1". The first completion token sits at position PROMPT_LEN,
# and it is scored by the logits at position PROMPT_LEN - 1. So the slice
# starts at PROMPT_LEN - 1. Slicing at PROMPT_LEN would silently drop the
# first generated token from the loss.
# ---------------------------------------------------------------------------
def part3_prompt_mask(model):
    print("=" * 72)
    print("PART 3  why selected[prompt_len - 1:], not selected[prompt_len:]")
    print("=" * 72)

    logits = model(TOKEN_IDS.unsqueeze(0)).squeeze(0).float()
    logprobs = torch.log_softmax(logits, dim=-1)
    selected = logprobs[:-1].gather(1, TOKEN_IDS[1:].unsqueeze(-1)).squeeze(-1)

    print(f"  prompt = {VOCAB[:PROMPT_LEN]}  (prompt_len={PROMPT_LEN})")
    print(f"  completion = {VOCAB[PROMPT_LEN:]}\n")

    print(f"  {'j':>3}  {'selected[j] scores':<22} {'value':>9}  keep?")
    for j in range(len(selected)):
        label = f"{VOCAB[TOKEN_IDS[j]]!r} -> {VOCAB[TOKEN_IDS[j + 1]]!r}"
        keep = "KEEP" if j >= PROMPT_LEN - 1 else "prompt, drop"
        print(f"  {j:>3}  {label:<22} {selected[j].item():>9.4f}  {keep}")

    correct = selected[PROMPT_LEN - 1:]
    off_by_one = selected[PROMPT_LEN:]
    print(f"\n  selected[{PROMPT_LEN - 1}:] -> {correct.numel()} values, "
          f"sum={correct.sum().item():.4f}   <- correct")
    print(f"  selected[{PROMPT_LEN}:]  -> {off_by_one.numel()} values, "
          f"sum={off_by_one.sum().item():.4f}   <- loses "
          f"{VOCAB[TOKEN_IDS[PROMPT_LEN]]!r}, the first generated token")

    real = sequence_logprob(model, TOKEN_IDS, PROMPT_LEN)
    draft = sequence_logprob_draft(model, TOKEN_IDS, PROMPT_LEN)
    print(f"\n  trials_00.sequence_logprob      : {real.item():.4f}")
    print(f"  trials_00.sequence_logprob_draft: {draft.item():.4f}")
    print(f"  match our hand-built sum        : "
          f"{torch.allclose(real, correct.sum())}\n")


# ---------------------------------------------------------------------------
# Part 4 -- sum, not mean.
#
# ch05 used a length-normalized mean (avg_logprob_answer) because it was
# COMPARING candidate answers of different lengths. GRPO instead needs
# log p(sequence) = sum of per-token logprobs, the actual quantity whose
# gradient is the policy gradient. Dividing by length would rescale each
# rollout's gradient by 1/len, quietly down-weighting long rollouts.
# ---------------------------------------------------------------------------
def part4_sum_not_mean(model):
    print("=" * 72)
    print("PART 4  sum vs. mean")
    print("=" * 72)

    logits = model(TOKEN_IDS.unsqueeze(0)).squeeze(0).float()
    logprobs = torch.log_softmax(logits, dim=-1)
    selected = logprobs[:-1].gather(
        1, TOKEN_IDS[1:].unsqueeze(-1)
    ).squeeze(-1)[PROMPT_LEN - 1:]

    total = selected.sum()
    print(f"  per-token logprobs : {[f'{v:.4f}' for v in selected.tolist()]}")
    print(f"  sum  (GRPO uses)   : {total.item():.4f}")
    print(f"  mean (ch05 used)   : {selected.mean().item():.4f}")
    print(f"  exp(sum) = p(seq)  : {total.exp().item():.6f}")
    print("  -> logprobs are negative, so longer sequences score lower;")
    print("     that is correct, a longer sequence really is less likely\n")

    grad_target = -total
    grad_target.backward()
    print(f"  d(-logp)/d(out_head.weight) norm: "
          f"{model.out_head.weight.grad.norm().item():.4f}")
    print("  -> this is the tensor GRPO scales by the advantage")
    print("     (trials_00.py:183, pg_loss = -(advantages.detach() * logps).mean())")


def main():
    torch.manual_seed(123)
    model = ToyModel(V)

    part1_the_shift(model)
    part2_three_spellings(model)
    part3_prompt_mask(model)
    part4_sum_not_mean(model)


if __name__ == "__main__":
    main()
