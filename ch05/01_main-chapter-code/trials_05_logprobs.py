# Concept: sequence log-probabilities (notebook cells 58-61)   [loads model]
#
# Products become sums in log space, avoiding near-zero joint probabilities.
# Less-negative totals indicate sequences the model considers more likely.

import torch

from trials_00 import calc_next_token_logprobas, load_base_model


def main():
    model, tokenizer, device = load_base_model()
    print("Device:", device)
    torch.set_printoptions(precision=4, sci_mode=False)
    for prompt in (
        "The capital of Germany is Berlin",
        "The capital of Germany is Bridge",
        "The capital of Germany is Hamburg",
    ):
        print(f"\n{prompt!r}")
        calc_next_token_logprobas(model, tokenizer, prompt, device)


if __name__ == "__main__":
    main()