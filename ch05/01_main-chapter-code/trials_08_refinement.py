# Concept: iterative self-refinement loop (notebook cells 83-86)   [loads model]
#
# partial binds model context into the scorer so the loop supplies only answer
# and prompt. Better score does not necessarily mean mathematical correctness.

from functools import partial

import torch

from trials_00 import (
    RAW_PROMPT,
    avg_logprob_answer,
    load_base_model,
    self_refinement_loop,
)


def main():
    model, tokenizer, device = load_base_model()
    print("Device:", device)
    avg_logprob_score = partial(
        avg_logprob_answer,
        model=model,
        tokenizer=tokenizer,
        device=device,
    )

    torch.manual_seed(1)
    results = self_refinement_loop(
        model=model,
        tokenizer=tokenizer,
        raw_prompt=RAW_PROMPT,
        device=device,
        iterations=2,
        max_response_tokens=2048,
        max_critique_tokens=256,
        score_fn=avg_logprob_score,
        verbose=True,
        temperature=0.7,
        top_p=0.9,
    )
    print(results["final_extracted"])


if __name__ == "__main__":
    main()