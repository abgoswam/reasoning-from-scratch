# Concept: <think> tokens (notebook cells 130-145, section 7.6.1)
#
# Before you can reward a model for writing <think> ... </think>, the tags
# have to BE tokens. Whether they are depends on which tokenizer you loaded,
# and the chapter's answer is to switch models rather than patch the base
# tokenizer. This file shows why.
#
# Downloads the two Qwen3 tokenizer files (~18 MB total, cached afterwards).
# No model weights are loaded -- the vocabulary size comes from the config
# dict, not from the checkpoint.
#
#   Part 1: the base tokenizer shreds <think> into three tokens
#   Part 2: patching it with add_special_tokens
#   Part 3: the reasoning tokenizer already has them
#   Part 4: the embedding table already has rows for them
#   Part 5: why 7_6 trains the reasoning model

from pathlib import Path

from reasoning_from_scratch.qwen3 import (
    QWEN_CONFIG_06_B,
    Qwen3Tokenizer,
    download_qwen3_small,
)

THINK_TOKEN_ID = 151667
END_THINK_TOKEN_ID = 151668
LOCAL_DIR = Path(__file__).resolve().parent / "qwen3"


def load(kind):
    download_qwen3_small(kind=kind, tokenizer_only=True, out_dir=LOCAL_DIR)
    return Qwen3Tokenizer(
        tokenizer_file_path=LOCAL_DIR / f"tokenizer-{kind}.json"
    )


# ---------------------------------------------------------------------------
# Part 1 -- to the base tokenizer, "<think>" is just five characters.
#
# It has no entry for the tag, so BPE splits it into whatever pieces it does
# know. The reward function looks for a single id 151667 and will never find
# it, so reward_format would return 0.0 for every rollout forever.
# ---------------------------------------------------------------------------
def part1_base_shreds_it(tok_base):
    print("=" * 74)
    print("PART 1  the base tokenizer has no <think> token")
    print("=" * 74)

    for text in ("<think>", "</think>"):
        ids = tok_base.encode(text)
        pieces = [tok_base.decode([i]) for i in ids]
        print(f"  encode({text!r:<10}) -> {ids}")
        print(f"  {'':<20}    pieces: {pieces}")

    ids = tok_base.encode("<think>")
    print(f"\n  looking for id {THINK_TOKEN_ID} in {ids}: "
          f"{THINK_TOKEN_ID in ids}")
    print("\n  -> reward_format would score 0.0 on every rollout, and the")
    print("     model would never learn the format because the signal is")
    print("     constant. A dead reward is worse than no reward.\n")
    return ids


# ---------------------------------------------------------------------------
# Part 2 -- the tokenizer can be patched.
#
# add_special_tokens teaches it to treat the tag as one unit. Note this
# mutates the tokenizer in place, and note WHICH ids it hands out.
# ---------------------------------------------------------------------------
def part2_patch_it(tok_base, before):
    print("=" * 74)
    print("PART 2  add_special_tokens")
    print("=" * 74)

    tok_base._tok.add_special_tokens(
        ["<tool_response>", "</tool_response>", "<think>", "</think>"]
    )
    after_open = tok_base.encode("<think>")
    after_close = tok_base.encode("</think>")

    print(f"  before : {before}")
    print(f"  after  : {after_open}   <think>")
    print(f"  after  : {after_close}   </think>")
    print(f"\n  do these match the constants the reward function uses?")
    print(f"    <think>  == {THINK_TOKEN_ID}? {after_open == [THINK_TOKEN_ID]}")
    print(f"    </think> == {END_THINK_TOKEN_ID}? "
          f"{after_close == [END_THINK_TOKEN_ID]}")
    print("\n  -> the tags are now single tokens. But a token id is only half")
    print("     the story: the MODEL has to have learned something about that")
    print("     id, and a base model has never seen these tags in training.\n")


# ---------------------------------------------------------------------------
# Part 3 -- the reasoning tokenizer needs no patching.
# ---------------------------------------------------------------------------
def part3_reasoning_tokenizer(tok_reasoning):
    print("=" * 74)
    print("PART 3  the reasoning tokenizer, unmodified")
    print("=" * 74)

    open_ids = tok_reasoning.encode("<think>")
    close_ids = tok_reasoning.encode("</think>")
    print(f"  encode('<think>')  -> {open_ids}")
    print(f"  encode('</think>') -> {close_ids}")
    print(f"\n  matches THINK_TOKEN_ID     = {THINK_TOKEN_ID}: "
          f"{open_ids == [THINK_TOKEN_ID]}")
    print(f"  matches END_THINK_TOKEN_ID = {END_THINK_TOKEN_ID}: "
          f"{close_ids == [END_THINK_TOKEN_ID]}")
    print("\n  -> these are the two constants at the top of ch07.py and of")
    print("     7_6_plus_format_reward.py. They are not arbitrary; they are")
    print("     what this tokenizer already assigns.\n")


# ---------------------------------------------------------------------------
# Part 4 -- the embedding table is big enough either way.
#
# The notebook loads the whole model just to print tok_emb.weight.shape[0].
# That number is in the config dict, so no weights are needed.
# ---------------------------------------------------------------------------
def part4_embeddings_exist():
    print("=" * 74)
    print("PART 4  the ids already have embedding rows")
    print("=" * 74)

    vocab = QWEN_CONFIG_06_B["vocab_size"]
    print(f"  QWEN_CONFIG_06_B['vocab_size'] : {vocab:,}")
    print(f"  THINK_TOKEN_ID                 : {THINK_TOKEN_ID:,}")
    print(f"  END_THINK_TOKEN_ID             : {END_THINK_TOKEN_ID:,}")
    print(f"\n  both ids < vocab_size: "
          f"{max(THINK_TOKEN_ID, END_THINK_TOKEN_ID) < vocab}")
    print(f"  spare rows above them: {vocab - END_THINK_TOKEN_ID - 1:,}")
    print("\n  -> no resizing needed. The base and reasoning models share the")
    print("     same 151,936-row table; the difference is whether those two")
    print("     rows were ever TRAINED, not whether they exist.\n")


# ---------------------------------------------------------------------------
# Part 5 -- so why does 7_6 switch models?
#
# You can give the base model the tokens (part 2) but not the habit. Its
# embedding rows for 151667/151668 are effectively random, it has never
# emitted them, and reward_format only pays out once they appear -- which
# requires stumbling onto them by chance first. The reasoning model was
# already trained to use them, so the format reward starts from a signal
# that is sometimes 1 instead of always 0.
# ---------------------------------------------------------------------------
def part5_why_switch_models():
    print("=" * 74)
    print("PART 5  why 7_6 trains the reasoning model")
    print("=" * 74)

    print("  7_3, 7_4, 7_5, olmo3, deepseek_v32 -> which_model='base'")
    print("  7_6, gdpo                          -> which_model='reasoning'")
    print()
    print("  The reward is only informative once the tags actually appear.")
    print("  A base model has to produce a 2-token sequence it has never")
    print("  emitted, by chance, inside a 512-token sample, before the group")
    print("  sees any variation -- and with no variation, adv_std is 0 and")
    print("  the step teaches nothing (trials_02 part 3).")
    print()
    print("  Practical consequence: the 7.6 reference run starts at 50.8%")
    print("  MATH-500 accuracy where the others start at 15.6%. That gap is")
    print("  the model swap, not the format reward. 7.5 -> 7.6 is NOT an")
    print("  apples-to-apples comparison, and neither will your runs be.\n")


def main():
    tok_base = load("base")
    before = part1_base_shreds_it(tok_base)
    part2_patch_it(tok_base, before)
    tok_reasoning = load("reasoning")
    part3_reasoning_tokenizer(tok_reasoning)
    part4_embeddings_exist()
    part5_why_switch_models()


if __name__ == "__main__":
    main()
