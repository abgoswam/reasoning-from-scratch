# Concept: the policy-gradient loss (notebook cells 85-86)      [loads model]
#
#   pg_loss = -(advantages.detach() * logps).mean()
#
# .detach() on the advantages is the load-bearing part: they are a fixed
# learning signal, so gradients flow only through logps. Dropping it would
# backprop into the reward path, which is not what GRPO does.

import torch

from reasoning_from_scratch.ch03 import render_prompt
from trials_00 import (
    RAW_PROMPT, ROLLOUTS, load_base_model, reward_rlvr, sample_response,
)
from trials_05_advantages import advantages_from
from trials_06_logprobs import rollout_logprobs


def main():
    model, tokenizer, device = load_base_model()
    prompt = render_prompt(RAW_PROMPT)

    torch.manual_seed(0)
    _, prompt_len, _ = sample_response(
        model=model, tokenizer=tokenizer, prompt=prompt, device=device,
        max_new_tokens=512, temperature=0.9, top_p=0.9,
    )

    rewards = torch.tensor(
        [reward_rlvr(answer_text=a, ground_truth="83") for a in ROLLOUTS],
        device=device,
    )
    advantages = advantages_from(rewards)
    logps = torch.stack(
        rollout_logprobs(model, tokenizer, prompt, prompt_len, device)
    )

    print("advantages:", advantages)
    print("logps:     ", logps)

    pg_loss = -(advantages.detach() * logps).mean()
    print("pg_loss:   ", pg_loss)


if __name__ == "__main__":
    main()
