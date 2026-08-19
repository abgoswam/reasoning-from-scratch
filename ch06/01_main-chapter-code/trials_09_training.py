# Concept: the RLVR training loop (notebook cells 98-100)  [model, very slow]
#
# Wraps compute_grpo_loss in an otherwise ordinary training loop: optimizer,
# backward, step, checkpoint. Only stage 4 is GRPO-specific.
#
# Defaults here are deliberately smaller than the notebook's (steps=50,
# num_rollouts=4, max_new_tokens=512) so a debug run finishes. Raise them for
# a real run, and expect hours on CPU.

import torch

from trials_00 import (
    load_base_model, load_math_train, pick_cuda_device, train_rlvr_grpo,
)


def main(steps=2, num_rollouts=2, max_new_tokens=128, device=None,
         mem_debug=False):
    model, tokenizer, _ = load_base_model()
    math_train = load_math_train()

    if device is None:
        device = pick_cuda_device()
    model.to(device)
    print("training on:", device)

    torch.manual_seed(0)
    train_rlvr_grpo(
        model=model,
        tokenizer=tokenizer,
        math_data=math_train,
        device=device,
        steps=steps,
        num_rollouts=num_rollouts,
        max_new_tokens=max_new_tokens,
        temperature=0.8,
        top_p=0.9,
        lr=1e-5,
        checkpoint_every=5,
        checkpoint_dir=".",
        csv_log_path="train_rlvr_grpo_metrics.csv",
        mem_debug=mem_debug,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default=None,
                        help="e.g. cuda:0 -- pin a GPU instead of picking the freest")
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--num-rollouts", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--mem-debug", action="store_true",
                        help="print CUDA memory at each stage boundary")
    args = parser.parse_args()

    main(
        steps=args.steps,
        num_rollouts=args.num_rollouts,
        max_new_tokens=args.max_new_tokens,
        device=torch.device(args.device) if args.device else None,
        mem_debug=args.mem_debug,
    )
