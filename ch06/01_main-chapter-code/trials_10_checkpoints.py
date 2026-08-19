# Concept: the author's trained checkpoints (notebook cell 107)  [downloads]
#
# Skips the training cost: fetches a checkpoint the author already trained, so
# the post-GRPO model can be evaluated without running trials_09 to the end.
# Load it into a Qwen3Model with model.load_state_dict(torch.load(path)).

from reasoning_from_scratch.qwen3 import download_qwen3_grpo_checkpoints


def main(grpo_type="no_kl", step="00050"):
    download_qwen3_grpo_checkpoints(grpo_type=grpo_type, step=step)
    print(f"downloaded grpo_type={grpo_type} step={step}")


if __name__ == "__main__":
    main()
