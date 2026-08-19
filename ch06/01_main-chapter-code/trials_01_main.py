# Concept: setup and a baseline sample (notebook cells 32-33)   [loads model]
#
# The starting point RLVR trains away from: the untuned base model, sampled
# stochastically. The answer is 83; the base model will usually miss it.
# Output will not match the book -- see the bfloat16/CPU-kernel note.

import torch

from reasoning_from_scratch.ch03 import render_prompt
from reasoning_from_scratch.ch04 import (
    generate_text_stream_concat_flex,
    generate_text_top_p_stream_cache,
)
from trials_00 import RAW_PROMPT, load_base_model


def main():
    model, tokenizer, device = load_base_model()

    prompt = render_prompt(RAW_PROMPT)
    print("--- rendered prompt ---")
    print(prompt)
    print("--- sampled response ---")

    torch.manual_seed(0)
    response = generate_text_stream_concat_flex(
        model, tokenizer, prompt, device,
        max_new_tokens=2048, verbose=True,
        generate_func=generate_text_top_p_stream_cache,
        temperature=0.9,
        top_p=0.9,
    )
    print()
    print("response =", repr(response))
    print("correct answer = 83")


if __name__ == "__main__":
    main()
