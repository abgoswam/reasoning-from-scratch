# Concept: next-token probabilities (notebook cells 38-45)   [loads model]
#
# Multiplying token probabilities across a sequence quickly approaches zero.
# Device-specific floating-point kernels can change the final tiny product.

import torch

from trials_00 import calc_next_token_probas, load_base_model


def main():
    model, tokenizer, device = load_base_model()
    print("Device:", device)
    torch.set_printoptions(precision=4, sci_mode=True)
    for prompt in (
        "The capital of Germany is Berlin",
        "The capital of Germany is Bridge",
        "The capital of Germany is Hamburg",
    ):
        print(f"\n{prompt!r}")
        calc_next_token_probas(model, tokenizer, prompt, device)


if __name__ == "__main__":
    main()