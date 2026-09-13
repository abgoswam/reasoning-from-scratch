"""Stagewise tutorial on PyTorch advanced ("fancy") indexing.

Run with: python tutorial_indexing.py
Each stage prints what happens and asserts the expected result,
building up to the exact pattern used in calc_next_token_logprobas:
    all_logprobas[t_idx, next_ids]
"""

import torch


def stage1_basic_indexing():
    print("\n=== Stage 1: basic (non-fancy) indexing ===")
    t = torch.arange(10)
    print("t            :", t)
    print("t[3]         :", t[3].item())          # single element
    print("t[2:5]       :", t[2:5])                # slice

    assert t[3].item() == 3
    assert torch.equal(t[2:5], torch.tensor([2, 3, 4]))
    print("OK: plain indexing/slicing behave as expected")


def stage2_fancy_indexing_1d():
    print("\n=== Stage 2: fancy indexing with an index tensor (1D) ===")
    t = torch.arange(10) * 10  # [0, 10, 20, ..., 90]
    idx = torch.tensor([1, 3, 5])
    picked = t[idx]
    print("t            :", t)
    print("idx          :", idx)
    print("t[idx]       :", picked)

    # Fancy indexing gathers elements at each given position (not a slice)
    assert torch.equal(picked, torch.tensor([10, 30, 50]))
    print("OK: t[idx] == [t[1], t[3], t[5]]")


def stage3_2d_row_col_selection():
    print("\n=== Stage 3: 2D indexing - selecting whole rows/cols ===")
    mat = torch.arange(12).reshape(3, 4)
    print("mat:\n", mat)

    rows = mat[[0, 2]]        # select rows 0 and 2 (each a full row)
    cols = mat[:, [1, 3]]     # select columns 1 and 3 (each a full column)
    print("mat[[0, 2]]      :\n", rows)
    print("mat[:, [1, 3]]   :\n", cols)

    assert torch.equal(rows, torch.stack([mat[0], mat[2]]))
    assert torch.equal(cols, torch.stack([mat[:, 1], mat[:, 3]], dim=1))
    print("OK: list/tensor index on ONE axis selects whole rows or columns")


def stage4_paired_fancy_indexing():
    print("\n=== Stage 4: paired fancy indexing (the key mechanism) ===")
    mat = torch.arange(12).reshape(3, 4)
    print("mat:\n", mat)

    row_idx = torch.tensor([0, 1, 2])
    col_idx = torch.tensor([3, 0, 1])
    picked = mat[row_idx, col_idx]
    print("row_idx           :", row_idx)
    print("col_idx           :", col_idx)
    print("mat[row_idx, col_idx]:", picked)

    # IMPORTANT: this is NOT a cross product (that would give 9 values).
    # It pairs elements position-by-position:
    #   (row_idx[0], col_idx[0]) -> mat[0, 3]
    #   (row_idx[1], col_idx[1]) -> mat[1, 0]
    #   (row_idx[2], col_idx[2]) -> mat[2, 1]
    manual = torch.tensor([mat[0, 3], mat[1, 0], mat[2, 1]])
    assert torch.equal(picked, manual)
    assert picked.shape == (3,)
    print("OK: mat[rows, cols] pairs up rows[i] with cols[i] elementwise")
    print("    result shape == rows.shape == cols.shape, NOT (len(rows), len(cols))")


def stage5_broadcasting_in_fancy_indexing():
    print("\n=== Stage 5: broadcasting turns pairing into a cross product ===")
    mat = torch.arange(12).reshape(3, 4)

    row_idx = torch.tensor([0, 1, 2]).unsqueeze(1)  # shape (3, 1)
    col_idx = torch.tensor([0, 2]).unsqueeze(0)      # shape (1, 2)
    picked = mat[row_idx, col_idx]                   # broadcasts to (3, 2)
    print("row_idx shape:", row_idx.shape, "col_idx shape:", col_idx.shape)
    print("mat[row_idx, col_idx]:\n", picked)

    expected = torch.stack([mat[:, 0], mat[:, 2]], dim=1)
    assert torch.equal(picked, expected)
    print("OK: when shapes broadcast instead of matching directly, you DO get")
    print("    a cross-product-like result. This is why shapes matter!")


def stage6_apply_to_logprobas():
    print("\n=== Stage 6: the real use case - gathering next-token log-probs ===")
    torch.manual_seed(0)
    seq_len, vocab_size = 6, 20  # small vocab for readability
    token_ids = torch.randint(0, vocab_size, (seq_len,))
    logits = torch.randn(seq_len, vocab_size)
    all_logprobas = torch.log_softmax(logits, dim=-1)
    print("token_ids       :", token_ids)
    print("all_logprobas.shape:", all_logprobas.shape)

    t_idx = torch.arange(0, token_ids.shape[0] - 1)  # [0, 1, 2, 3, 4]
    next_ids = token_ids[1:]                          # 5 "true" next tokens
    next_token_logprobas = all_logprobas[t_idx, next_ids]
    print("t_idx           :", t_idx)
    print("next_ids        :", next_ids)
    print("next_token_logprobas:", next_token_logprobas)

    # Manual/explicit version to prove the pairing interpretation
    manual = torch.stack(
        [all_logprobas[i, next_ids[i]] for i in range(t_idx.shape[0])]
    )
    assert torch.equal(next_token_logprobas, manual)
    assert next_token_logprobas.shape == (seq_len - 1,)

    # Equivalent using torch.gather, for comparison
    gathered = all_logprobas.gather(1, next_ids.unsqueeze(1)).squeeze(1)
    assert torch.allclose(next_token_logprobas, gathered)
    print("OK: all_logprobas[t_idx, next_ids] == per-position log-prob of the")
    print("    actual next token, same result as torch.gather")


if __name__ == "__main__":
    stage1_basic_indexing()
    stage2_fancy_indexing_1d()
    stage3_2d_row_col_selection()
    stage4_paired_fancy_indexing()
    stage5_broadcasting_in_fancy_indexing()
    stage6_apply_to_logprobas()
    print("\nAll stages passed.")
