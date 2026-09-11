# Concept: critique and refinement prompts (notebook cells 78-80)   [loads model]
#
# The same model acts as solver, critic, and reviser. Each stage starts a fresh
# generation carrying the prior text through its prompt.

import torch

from reasoning_from_scratch.ch04 import (
    generate_text_stream_concat_flex,
    generate_text_top_p_stream_cache,
)
from trials_00 import (
    PROMPT,
    RAW_PROMPT,
    load_base_model,
    make_critique_prompt,
    make_refine_prompt,
)


def generate(model, tokenizer, prompt, device):
    return generate_text_stream_concat_flex(
        model, tokenizer, prompt, device,
        max_new_tokens=2048, verbose=True,
        generate_func=generate_text_top_p_stream_cache,
        temperature=0.7, top_p=0.9,
    )


def main():
    model, tokenizer, device = load_base_model()
    print("Device:", device)
    torch.manual_seed(123)
    initial_response = generate(model, tokenizer, PROMPT, device)
    torch.manual_seed(123)
    critique = generate(
        model, tokenizer,
        make_critique_prompt(RAW_PROMPT, initial_response), device,
    )
    torch.manual_seed(123)
    revised_answer = generate(
        model, tokenizer,
        make_refine_prompt(RAW_PROMPT, initial_response, critique), device,
    )
    print("\nInitial response:\n", initial_response)
    print("\nCritique:\n", critique)
    print("\nRevised answer:\n", revised_answer)


if __name__ == "__main__":
    main()