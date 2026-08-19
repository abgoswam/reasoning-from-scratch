# Concept: one full GRPO step, assembled (notebook cells 91-93) [loads model]
#
# compute_grpo_loss is stages 1-4 of the figure in one call: sample a group of
# rollouts, score them, normalize into advantages, weight the sequence
# logprobs. Everything the earlier trials_XX scripts did piecewise.
# This is the function worth setting breakpoints in.

from pprint import pprint

import torch

from trials_00 import compute_grpo_loss, load_base_model, load_math_train


def main():
    model, tokenizer, device = load_base_model()
    math_train = load_math_train()

    print("example:")
    pprint(math_train[4])
    print()

    torch.manual_seed(123)
    stats = compute_grpo_loss(
        model=model,
        tokenizer=tokenizer,
        example=math_train[4],
        device=device,
        num_rollouts=2,
        max_new_tokens=256,
        temperature=0.8,
        top_p=0.9,
    )
    pprint(stats)


if __name__ == "__main__":
    main()
