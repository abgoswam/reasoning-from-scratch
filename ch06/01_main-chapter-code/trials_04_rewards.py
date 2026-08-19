# Concept: verifiable rewards (notebook cells 52, 56-57)
#
# The "VR" in RLVR. No learned reward model: extract \boxed{...} and grade it
# against ground truth. Note rollout 3 scores 0 despite being correct -- the
# fallback=None argument means an answer without \boxed{} earns nothing.

from trials_00 import ROLLOUTS, reward_rlvr


def main():
    rollout_rewards = []

    for answer in ROLLOUTS:
        reward = reward_rlvr(answer_text=answer, ground_truth="83")
        print(f"Answer: {answer!r}")
        print(f"Reward: {reward}\n")
        rollout_rewards.append(reward)

    print("rollout_rewards =", rollout_rewards)
    return rollout_rewards


if __name__ == "__main__":
    main()
