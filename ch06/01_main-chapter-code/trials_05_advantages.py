# Concept: group-relative advantages (notebook cells 64-66)
#
# The "GR" in GRPO. Rewards are centered and scaled within the rollout group,
# so the signal says "better or worse than its peers", not "good or bad".
# When every rollout in a group scores the same, all advantages collapse to
# zero and the step contributes no gradient -- worth seeing directly.

import torch

from trials_00 import ROLLOUTS, reward_rlvr


def advantages_from(rewards):
    return (rewards - rewards.mean()) / (rewards.std() + 1e-4)


def main():
    device = torch.device("cpu")

    rollout_rewards = [
        reward_rlvr(answer_text=a, ground_truth="83") for a in ROLLOUTS
    ]
    rewards = torch.tensor(rollout_rewards, device=device)
    print("rewards:   ", rewards)
    print("advantages:", advantages_from(rewards))

    print("\nDegenerate groups (no learning signal):")
    for label, vals in (("all correct", [1.0] * 4), ("all wrong", [0.0] * 4)):
        r = torch.tensor(vals, device=device)
        print(f"  {label:12s} -> {advantages_from(r)}")


if __name__ == "__main__":
    main()
