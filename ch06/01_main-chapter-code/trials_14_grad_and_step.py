# Concept: who gets a .grad, and what optimizer.step() does with it
#
# Toy tensors only -- no model, no GPU, instant. Every number is checkable
# by hand, so put a breakpoint anywhere and inspect.
#
#   Part 1: .grad exists only on LEAF tensors with requires_grad=True
#   Part 2: the graph autograd built, node by node
#   Part 3: .grad ACCUMULATES -- why zero_grad() exists
#   Part 4: optimizer.step() is just "read .grad, write to .data"
#   Part 5: the same three calls in trials_00.py's training loop
#
# The toy function, chosen so the gradients are obvious:
#
#     y = (w * x).sum() + b
#     dy/dw = x        dy/db = 1        dy/dx = w
#
# Written one operation at a time, so every intermediate has a name you can
# inspect in the debugger:
#
#     tmp1 = w * x        [3., 8.]     MulBackward0
#     tmp2 = tmp1.sum()   11.0         SumBackward0
#     tmp3 = tmp2 + b     11.5         AddBackward0     <- this IS y

import torch


def g(tensor):
    # Reading .grad on a non-leaf emits a UserWarning; check first so the
    # tables below stay clean. `retains_grad` is True after retain_grad().
    if not (tensor.is_leaf or tensor.retains_grad):
        return "None"
    return "None" if tensor.grad is None else str(tensor.grad.tolist())


# ---------------------------------------------------------------------------
# Part 1 -- two conditions, both required, for .grad to be filled in:
#
#   (a) is_leaf       -- you created it; it is not the output of an operation
#   (b) requires_grad -- you asked for gradients
#
# Note "learnable" is NOT the rule. x below is input data, but if you set
# requires_grad=True on it, it gets a .grad too. PyTorch has no concept of
# "parameter" at this level -- only leaf + requires_grad.
#
# Breaking the expression into tmp1/tmp2/tmp3 makes the pattern visible:
# the three tensors YOU typed are leaves, the three the ops produced are not.
# ---------------------------------------------------------------------------
def part1_who_gets_grad():
    print("=" * 74)
    print("PART 1  .grad lives on leaf tensors with requires_grad=True")
    print("=" * 74)

    x = torch.tensor([1.0, 2.0])                          # input data
    w = torch.tensor([3.0, 4.0], requires_grad=True)      # learnable
    b = torch.tensor(0.5, requires_grad=True)             # learnable

    tmp1 = w * x          # elementwise multiply
    tmp2 = tmp1.sum()     # reduce to a scalar
    tmp3 = tmp2 + b       # add the bias -- this IS y

    # retain_grad() is the ONLY reason the intermediates show a .grad below.
    # Comment these out and tmp1/tmp2/tmp3 all report None (see Part 2).
    for t in (tmp1, tmp2, tmp3):
        t.retain_grad()

    tmp3.backward()

    rows = [("x", x), ("w", w), ("b", b),
            ("tmp1", tmp1), ("tmp2", tmp2), ("tmp3 (=y)", tmp3)]
    print(f"  {'tensor':<11} {'value':>12} {'is_leaf':>8} {'req_grad':>9} "
          f"{'grad_fn':<15} {'.grad':>12}")
    print("  " + "-" * 72)
    for name, t in rows:
        fn = type(t.grad_fn).__name__ if t.grad_fn is not None else "None"
        print(f"  {name:<11} {str(t.tolist()):>12} {str(t.is_leaf):>8} "
              f"{str(t.requires_grad):>9} {fn:<15} {g(t):>12}")

    print("\n  is_leaf     -- the three YOU typed are leaves; the three the")
    print("                 operations produced are not")
    print("  req_grad    -- contagious forward: x is False, but tmp1 = w * x")
    print("                 is True, because one operand was enough")
    print("  grad_fn     -- the exact mirror of is_leaf: leaves have None,")
    print("                 non-leaves carry the node that differentiates them")
    print("  .grad       -- filled for w and b only; the intermediates show a")
    print("                 value here ONLY because of retain_grad()\n")

    print(f"  w.grad == x    -> {w.grad.tolist()}, because dy/dw = x")
    print("  b.grad == 1.0  -> addition passes its gradient through unchanged")
    print("  x.grad is None -> requires_grad=False, nobody asked for it\n")

    # Same graph, but now we DO want a gradient w.r.t. the input data.
    x2 = torch.tensor([1.0, 2.0], requires_grad=True)
    w2 = torch.tensor([3.0, 4.0], requires_grad=True)
    ((w2 * x2).sum()).backward()
    print(f"  with requires_grad=True on the input: x2.grad = {g(x2)}")
    print("  -> equals w2, because dy/dx = w. 'Input data' is not special.\n")


# ---------------------------------------------------------------------------
# Part 2 -- the graph autograd built while Part 1 ran forward.
#
# Every forward op leaves behind exactly one backward NODE. The graph is a
# mirror of the code:
#
#     tmp1 = w * x        ->  MulBackward0
#     tmp2 = tmp1.sum()   ->  SumBackward0
#     tmp3 = tmp2 + b     ->  AddBackward0
#
# `next_functions` is badly named: it does NOT mean "the next op forward".
# It means "where do I hand my gradient next" -- the edges pointing BACK
# toward the inputs. The tuple is positional, so next_functions[i] belongs to
# input i of the forward op, and a None slot means that input needs nothing.
#
# AccumulateGrad is the terminal node for a leaf that requires grad. It holds
# the real tensor in .variable and performs `p.grad += incoming`. Count the
# AccumulateGrad nodes and you have counted the tensors that end up with a
# .grad -- here, exactly two.
# ---------------------------------------------------------------------------
def walk_graph(node, depth=0):
    if node is None:
        return
    name = type(node).__name__
    extra = ""
    if name == "AccumulateGrad":
        extra = f"   -> writes .grad of the leaf {node.variable.tolist()}"
    print("  " + "    " * depth + f"{name}{extra}")
    for i, (nxt, _) in enumerate(node.next_functions):
        label = type(nxt).__name__ if nxt is not None else "None   (input needs no gradient)"
        print("  " + "    " * depth + f"  next_functions[{i}] = {label}")
        walk_graph(nxt, depth + 1)


def part2_the_graph():
    print("=" * 74)
    print("PART 2  the graph: one backward node per forward op")
    print("=" * 74)

    x = torch.tensor([1.0, 2.0])
    w = torch.tensor([3.0, 4.0], requires_grad=True)
    b = torch.tensor(0.5, requires_grad=True)

    tmp1 = w * x
    tmp2 = tmp1.sum()
    tmp3 = tmp2 + b

    walk_graph(tmp3.grad_fn)

    print(f"\n  AccumulateGrad.variable is w : "
          f"{tmp1.grad_fn.next_functions[0][0].variable is w}")
    print(f"  the b branch of AddBackward0 : "
          f"{tmp3.grad_fn.next_functions[1][0].variable is b}")
    print(f"  MulBackward0 slot [1] (for x): "
          f"{tmp1.grad_fn.next_functions[1][0]}")
    print("  -> no node exists for x at all. That is the structural reason")
    print("     x.grad is None: not computed-then-discarded, never created.\n")

    print("  gradient values flowing down the tree:")
    print("    AddBackward0  gets 1.0    -> 1.0   to SumBackward0")
    print("                              -> 1.0   to AccumulateGrad(b)")
    print("    SumBackward0  gets 1.0    -> [1,1] to MulBackward0  (broadcast)")
    print("    MulBackward0  gets [1,1]  -> [1,2] to AccumulateGrad(w)  (x * incoming)")
    print("                              -> slot [1] is None, nothing sent\n")

    # The intermediates' gradients ARE computed -- backward cannot reach w
    # without them -- but without retain_grad() they are dropped on use.
    tmp3.backward()
    print("  without retain_grad(), after backward():")
    for name, t in (("tmp1", tmp1), ("tmp2", tmp2), ("tmp3", tmp3)):
        print(f"    {name}.grad = {g(t)}")
    print(f"    w.grad    = {g(w)}   <- still computed correctly\n")


# ---------------------------------------------------------------------------
# Part 3 -- .grad is a running SUM, not an assignment.
#
# backward() does `p.grad += new_grad`. This is deliberate: it lets you split
# a batch across several backward calls. It also means a forgotten
# zero_grad() silently trains on stale gradients.
# ---------------------------------------------------------------------------
def part3_accumulation():
    print("=" * 68)
    print("PART 3  .grad accumulates across backward() calls")
    print("=" * 68)

    x = torch.tensor([1.0, 2.0])
    w = torch.tensor([3.0, 4.0], requires_grad=True)

    print(f"  before any backward : {g(w)}")
    for call in (1, 2, 3):
        (w * x).sum().backward()
        print(f"  after backward #{call}   : {g(w)}")

    w.grad = None
    print(f"  after w.grad = None : {g(w)}")
    print("  -> optimizer.zero_grad() does exactly this for every parameter\n")


# ---------------------------------------------------------------------------
# Part 4 -- what step() actually does.
#
# The optimizer holds REFERENCES to the very same tensor objects. It never
# touches the graph; it reads p.grad and writes p.data. Plain SGD is a
# one-liner, so we can reproduce it exactly by hand.
# ---------------------------------------------------------------------------
def part4_optimizer_step():
    print("=" * 68)
    print("PART 4  optimizer.step() reads .grad, writes .data")
    print("=" * 68)

    x = torch.tensor([1.0, 2.0])
    w = torch.tensor([3.0, 4.0], requires_grad=True)
    lr = 0.1

    optimizer = torch.optim.SGD([w], lr=lr)
    print(f"  optimizer holds the SAME object: "
          f"{optimizer.param_groups[0]['params'][0] is w}\n")

    (w * x).sum().backward()
    print(f"  w before step : {w.tolist()}")
    print(f"  w.grad        : {g(w)}")

    by_hand = (w - lr * w.grad).tolist()
    optimizer.step()
    print(f"  w after step  : {w.tolist()}")
    print(f"  w - lr*w.grad : {by_hand}   <- identical\n")

    print(f"  .grad SURVIVES step(): {g(w)}")
    optimizer.zero_grad()
    print(f"  after zero_grad()    : {g(w)}")
    print("  -> step() does not clear it; that is a separate call\n")

    # trials_00.py uses AdamW, which needs .grad plus per-parameter state.
    w2 = torch.tensor([3.0, 4.0], requires_grad=True)
    adamw = torch.optim.AdamW([w2], lr=lr)
    (w2 * x).sum().backward()
    adamw.step()
    state = adamw.state[w2]
    print(f"  AdamW state keys : {sorted(k for k in state)}")
    print(f"  exp_avg          : {state['exp_avg'].tolist()}")
    print(f"  w2 after step    : {[round(v, 4) for v in w2.tolist()]}")
    print("  -> AdamW keeps running averages, so the update is NOT")
    print("     lr*grad. But .grad is still its only input from the graph.\n")


# ---------------------------------------------------------------------------
# Part 5 -- the same three calls, in the real loop (trials_00.py:~300).
#
#     optimizer.zero_grad()                    # Part 3: clear .grad
#     stats = compute_grpo_loss(...)           # forward, builds the graph
#     stats["loss_tensor"].backward()          # Part 1: fill every .grad
#     torch.nn.utils.clip_grad_norm_(...)      # rescale .grad in place
#     optimizer.step()                         # Part 4: .grad -> .data
# ---------------------------------------------------------------------------
def part5_clipping():
    print("=" * 68)
    print("PART 5  clip_grad_norm_ sits between backward and step")
    print("=" * 68)

    w = torch.tensor([3.0, 4.0], requires_grad=True)
    (w * torch.tensor([30.0, 40.0])).sum().backward()

    print(f"  .grad before clip : {g(w)}  norm={w.grad.norm():.2f}")
    torch.nn.utils.clip_grad_norm_([w], max_norm=1.0)
    print(f"  .grad after clip  : "
          f"{[round(v, 4) for v in w.grad.tolist()]}  "
          f"norm={w.grad.norm():.2f}")
    print("  -> edits .grad IN PLACE, before the optimizer ever reads it.")
    print("     RLVR needs this: one lucky rollout can produce a huge")
    print("     advantage, and an unclipped step would wreck the model.")


def main():
    part1_who_gets_grad()
    part2_the_graph()
    part3_accumulation()
    part4_optimizer_step()
    part5_clipping()


if __name__ == "__main__":
    main()
