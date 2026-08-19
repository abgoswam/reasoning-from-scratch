# Concept: rollouts (notebook cells 47-50)                      [loads model]
#
# A rollout is one sampled completion. sample_response differs from the ch04
# generators in what it returns: the full prompt+answer token ids and the
# prompt length, which is exactly what sequence_logprob needs later.
# Two seeds are run to show the group diversity GRPO depends on.

import torch

from reasoning_from_scratch.ch03 import render_prompt
from trials_00 import RAW_PROMPT, load_base_model, sample_response


def main():
    model, tokenizer, device = load_base_model()
    prompt = render_prompt(RAW_PROMPT)

    for seed in (0, 5):
        torch.manual_seed(seed)
        token_ids, prompt_len, answer_text = sample_response(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            device=device,
            max_new_tokens=512,
            temperature=0.9,
            top_p=0.9,
        )
        print(f"--- seed {seed} ---")
        print("token_ids.shape:", tuple(token_ids.shape))
        print("prompt_len:     ", prompt_len)
        print("answer:         ", repr(answer_text))
        print()


if __name__ == "__main__":
    main()
