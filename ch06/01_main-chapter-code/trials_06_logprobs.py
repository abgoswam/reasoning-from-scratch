# Concept: sequence log-probabilities (notebook cells 70-81)    [loads model]
#
# The chapter arrives at sequence_logprob in three steps:
#   1. avg_logprob_answer  (ch05)  -- length-normalized mean
#   2. sequence_logprob_draft      -- same, but summed instead of averaged
#   3. sequence_logprob            -- the gather-based rewrite GRPO uses
# Steps 2 and 3 must agree; that equivalence is the point of the section.
# The sum (not the mean) is what GRPO needs, so length is not divided out.

import torch

from reasoning_from_scratch.ch03 import render_prompt
from trials_00 import (
    RAW_PROMPT,
    ROLLOUTS,
    avg_logprob_answer,
    load_base_model,
    sample_response,
    sequence_logprob,
    sequence_logprob_draft,
)


def rollout_logprobs(model, tokenizer, prompt, prompt_len, device):
    logps = []
    for text in ROLLOUTS:
        token_ids = tokenizer.encode(prompt + " " + text)
        logprob = sequence_logprob(
            model=model,
            token_ids=torch.tensor(token_ids, device=device),
            prompt_len=prompt_len,
        )
        print(f"Answer:  {text}")
        print(f"Logprob: {logprob.item():.4f}\n")
        logps.append(logprob)
    return logps


def main():
    model, tokenizer, device = load_base_model()
    prompt = render_prompt(RAW_PROMPT)

    torch.manual_seed(0)
    token_ids, prompt_len, answer_text = sample_response(
        model=model, tokenizer=tokenizer, prompt=prompt, device=device,
        max_new_tokens=512, temperature=0.9, top_p=0.9,
    )
    print("answer:", repr(answer_text), "\n")

    avg = avg_logprob_answer(
        model, tokenizer, prompt=prompt, answer=answer_text, device=device
    )
    n_answer_tokens = len(tokenizer.encode(answer_text))
    print("1. avg_logprob_answer      :", avg)
    print("   x num answer tokens     :", avg * n_answer_tokens)

    draft = sequence_logprob_draft(model, token_ids, prompt_len)
    gather = sequence_logprob(model, token_ids, prompt_len)
    print("2. sequence_logprob_draft  :", draft.item())
    print("3. sequence_logprob        :", gather.item())
    print("   2 and 3 agree           :", torch.allclose(draft, gather))

    print("\n--- per-rollout sequence logprobs ---")
    rollout_logprobs(model, tokenizer, prompt, prompt_len, device)


if __name__ == "__main__":
    main()
