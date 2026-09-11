# Concept: baseline response generation (notebook cells 13-17)   [loads model]
#
# Two seeds produce answers with different lengths and content even though all
# generation settings are identical. Later scoring methods compare these runs.

import torch

from reasoning_from_scratch.ch04 import (
    generate_text_stream_concat_flex,
    generate_text_top_p_stream_cache,
)
from trials_00 import PROMPT_COT, load_base_model


def generate_responses(model, tokenizer, device):
    responses = []
    for seed in (0, 3):
        torch.manual_seed(seed)
        responses.append(generate_text_stream_concat_flex(
            model, tokenizer, PROMPT_COT, device,
            max_new_tokens=2048, verbose=True,
            generate_func=generate_text_top_p_stream_cache,
            temperature=0.9, top_p=0.9,
        ))
    return responses


def main():
    model, tokenizer, device = load_base_model()
    print("Device:", device)
    response_1, response_2 = generate_responses(model, tokenizer, device)
    print("Response 1 characters:", len(response_1))
    print("Response 1 tokens:", len(tokenizer.encode(response_1)))
    print("\nResponse 2 characters:", len(response_2))
    print("Response 2 tokens:", len(tokenizer.encode(response_2)))
    print("\nCorrect answer: 83")


if __name__ == "__main__":
    main()