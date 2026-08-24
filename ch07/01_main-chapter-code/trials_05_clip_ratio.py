# Concept: clipped policy ratios (notebook cells 84-90, section 7.4.1)
#
# Chapter 6's loss trusts every rollout equally. Chapter 7 adds a brake: if
# the updated policy has moved far from the one that generated a rollout,
# clip that rollout's influence. This is the PPO trick applied at sequence
# level.
#
# Hardcoded reward and logprob vectors, straight from the notebook. No model,
# no GPU, instant.
#
#   Part 1: the ch06 loss, for reference
#   Part 2: the ratio, and why it is exp(difference)
#   Part 3: clamping, and min(unclipped, clipped)
#   Part 4: why clip_eps=10.0 means "off"
#   Part 5: the first inner epoch is plain policy gradient
#   Part 6: which switch actually turns the clip on

import torch

REWARDS = torch.tensor([1., 1., 0., 0.])
NEW_LOGPS = torch.tensor([-7.9243, -20.1546, -16.6130, -23.3677])
OLD_LOGPS = torch.tensor([-10.9243, -20.3546, -14.6130, -23.3677])
#                          ^ +3.0     ^ +0.2     ^ -2.0     ^ same


def advantages_from(rewards):
    return (rewards - rewards.mean()) / (rewards.std() + 1e-4)


# ---------------------------------------------------------------------------
# Part 1 -- where chapter 6 stopped.
# ---------------------------------------------------------------------------
def part1_ch06_loss():
    print("=" * 74)
    print("PART 1  the chapter 6 loss")
    print("=" * 74)

    adv = advantages_from(REWARDS)
    pg_loss = -(adv.detach() * NEW_LOGPS).mean()

    print(f"  rewards    : {REWARDS.tolist()}")
    print(f"  advantages : {[round(a, 3) for a in adv.tolist()]}")
    print(f"  logprobs   : {NEW_LOGPS.tolist()}")
    print(f"\n  pg_loss = -(adv * logp).mean() = {pg_loss.item():.4f}")
    print("\n  -> nothing here notices whether the policy has drifted since")
    print("     these rollouts were generated.\n")


# ---------------------------------------------------------------------------
# Part 2 -- the ratio.
#
#     ratio = pi_new(rollout) / pi_old(rollout)
#           = exp(log pi_new - log pi_old)
#
# We work in logs the whole way, so the division becomes a subtraction and
# only the final exp() brings it back to a probability ratio. Doing it as an
# actual division of probabilities would underflow instantly: these are
# sequence logprobs around -20, i.e. probabilities near 1e-9.
# ---------------------------------------------------------------------------
def part2_the_ratio():
    print("=" * 74)
    print("PART 2  ratio = exp(new - old)")
    print("=" * 74)

    log_ratio = NEW_LOGPS - OLD_LOGPS
    ratio = torch.exp(log_ratio)

    print(f"  {'i':>3} {'old_logp':>10} {'new_logp':>10} {'diff':>8} "
          f"{'ratio':>9}   meaning")
    print("  " + "-" * 66)
    meanings = ["new policy 20x MORE likely", "slightly more likely",
                "new policy 7x LESS likely", "unchanged"]
    for i in range(4):
        print(f"  {i:>3} {OLD_LOGPS[i]:>10.4f} {NEW_LOGPS[i]:>10.4f} "
              f"{log_ratio[i]:>+8.2f} {ratio[i]:>9.4f}   {meanings[i]}")

    print(f"\n  ratio == 1  -> policy unchanged for that rollout")
    print(f"  ratio  > 1  -> the update made this rollout more likely")
    print(f"  ratio  < 1  -> less likely")

    p_new = torch.exp(NEW_LOGPS[0]).item()
    print(f"\n  why logs: exp({NEW_LOGPS[0]:.2f}) = {p_new:.3e} -- dividing raw")
    print("  probabilities at this scale loses all precision.\n")


# ---------------------------------------------------------------------------
# Part 3 -- clip, then take the pessimistic branch.
#
#     obj = min( ratio * adv,  clamp(ratio, 1-eps, 1+eps) * adv )
#
# The min() is what makes it a brake rather than a rescale. For a positive
# advantage it caps how much you can be rewarded for drifting; for a negative
# advantage the min picks the MORE negative branch, so drifting away from a
# bad rollout is never rewarded either.
# ---------------------------------------------------------------------------
def part3_clipping(clip_eps=0.2):
    print("=" * 74)
    print(f"PART 3  min(unclipped, clipped)   with clip_eps={clip_eps}")
    print("=" * 74)

    adv = advantages_from(REWARDS).detach()
    ratio = torch.exp(NEW_LOGPS - OLD_LOGPS)
    clipped_ratio = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps)

    unclipped = ratio * adv
    clipped = clipped_ratio * adv
    obj = torch.minimum(unclipped, clipped)

    print(f"  {'i':>3} {'adv':>7} {'ratio':>9} {'clipped':>9} "
          f"{'r*A':>9} {'c*A':>9} {'min':>9}  bound?")
    print("  " + "-" * 70)
    for i in range(4):
        bound = "CLIPPED" if not torch.isclose(obj[i], unclipped[i]) else ""
        print(f"  {i:>3} {adv[i]:>+7.3f} {ratio[i]:>9.4f} "
              f"{clipped_ratio[i]:>9.4f} {unclipped[i]:>+9.3f} "
              f"{clipped[i]:>+9.3f} {obj[i]:>+9.3f}  {bound}")

    print(f"\n  clipped_pg_loss = -mean(obj) = {(-obj.mean()).item():.4f}")
    print(f"  policy_ratio (logged) = mean(ratio) = {ratio.mean().item():.4f}")
    print("\n  -> the logged policy_ratio is a DRIFT gauge. Near 1.0 means the")
    print("     update is staying close to the rollout policy; far from 1.0")
    print("     means large steps and the clip is doing real work.\n")


# ---------------------------------------------------------------------------
# Part 4 -- the default clip_eps is 10.0, which disables the clip.
#
# clamp(ratio, 1-10, 1+10) = clamp(ratio, -9, 11). A probability ratio is
# never negative and rarely above 11, so the bound never binds and
# min(r*A, r*A) = r*A. The mechanism is present but inert until you lower it.
# ---------------------------------------------------------------------------
def part4_eps_sweep():
    print("=" * 74)
    print("PART 4  how often the clip actually binds")
    print("=" * 74)

    adv = advantages_from(REWARDS).detach()
    ratio = torch.exp(NEW_LOGPS - OLD_LOGPS)

    print(f"  ratios: {[round(r, 3) for r in ratio.tolist()]}\n")
    print(f"  {'clip_eps':>9} {'bounds':>18} {'# clipped':>10} {'loss':>10}")
    print("  " + "-" * 52)
    for eps in (10.0, 1.0, 0.5, 0.2, 0.1):
        cr = torch.clamp(ratio, 1.0 - eps, 1.0 + eps)
        obj = torch.minimum(ratio * adv, cr * adv)
        n = int((~torch.isclose(obj, ratio * adv)).sum())
        print(f"  {eps:>9.1f} {f'[{1-eps:.1f}, {1+eps:.1f}]':>18} "
              f"{n:>10} {(-obj.mean()).item():>10.4f}")

    print("\n  -> at the shipped default of 10.0 nothing is ever clipped and")
    print("     the loss equals part 1's. PPO commonly uses 0.2.\n")


# ---------------------------------------------------------------------------
# Part 5 -- in the FIRST inner epoch the ratio is identically 1, and there
# the clipped loss has the SAME GRADIENT as chapter 6's.
#
# The two losses do not have the same value:
#     ch06:  -(adv * logp).mean()      ~ -12
#     PPO :  -(ratio * adv).mean()     ~ 0     because ratio==1 and adv is centered
#
# But value is not what training uses. Differentiate the ratio at the point
# where new == old:
#     d/dtheta exp(logp_new - logp_old) = ratio * d(logp_new)/dtheta
#                                       =   1   * d(logp_new)/dtheta
# which is exactly chapter 6's gradient. So the first inner epoch of PPO IS
# vanilla policy gradient, and the clip cannot bind because every ratio is 1.
# Part 6 covers what happens in the epochs after that.
# ---------------------------------------------------------------------------
def part5_onpolicy_is_identity():
    print("=" * 74)
    print("PART 5  inner_epochs=1: same gradient, different value")
    print("=" * 74)

    adv = advantages_from(REWARDS).detach()

    # A stand-in for the model parameters: logp = base + theta, so theta=0 is
    # "the policy that generated the rollouts".
    theta_a = torch.zeros(4, requires_grad=True)
    loss_ch06 = -(adv * (NEW_LOGPS + theta_a)).mean()
    loss_ch06.backward()

    theta_b = torch.zeros(4, requires_grad=True)
    new_logps = NEW_LOGPS + theta_b
    old_logps = NEW_LOGPS.detach()          # frozen rollout policy
    ratio = torch.exp(new_logps - old_logps)
    obj = torch.minimum(ratio * adv, torch.clamp(ratio, 0.8, 1.2) * adv)
    loss_ppo = -obj.mean()
    loss_ppo.backward()

    print(f"  ratios                : {[round(r, 4) for r in ratio.tolist()]}")
    print()
    print(f"  ch06 loss VALUE       : {loss_ch06.item():>9.4f}")
    print(f"  PPO  loss VALUE       : {loss_ppo.item():>9.4f}   <- different")
    print()
    print(f"  ch06 GRADIENT         : {[round(g, 5) for g in theta_a.grad.tolist()]}")
    print(f"  PPO  GRADIENT         : {[round(g, 5) for g in theta_b.grad.tolist()]}")
    print(f"  identical             : "
          f"{torch.allclose(theta_a.grad, theta_b.grad)}   <- what matters")

    print("\n  -> the loss numbers in the two CSVs are not comparable, but the")
    print("     training is identical while ratio==1.\n")


# ---------------------------------------------------------------------------
# Part 6 -- so when does the clip ever DO anything?
#
# Only one thing makes the ratio leave 1: the model being updated while
# old_logps stay fixed. In 7_4_plus_clip_ratio.py that is inner_epochs.
#
#   old_model = copy.deepcopy(model)     # frozen snapshot, once per step
#   for _ in range(inner_epochs):
#       ... compute_grpo_loss(model=model, old_model=old_model, ...)
#       loss.backward(); optimizer.step()      # model moves, old_model does not
#
# Epoch 1 runs with model == old_model, so ratio == 1 and the clip is inert.
# Epoch 2 runs AFTER an optimizer step, so the ratio is genuinely different
# and the clip can bind. That is the whole mechanism.
#
# Both defaults matter, and they are not what you might guess:
#
#   7_4  --inner_epochs  default 2     <- the ratio DOES diverge by default
#   7_4  --clip_eps      default 10.0  <- but clamp(r, -9, 11) never binds
#   7_6  --inner_epochs  default 1     <- differs from 7_4; check per script
#
# So the shipped 7_4 run is not identical to 7_3: it takes two optimizer steps
# per training step against a frozen snapshot. What it does NOT do is clip.
# Lower clip_eps and the mechanism switches on.
# ---------------------------------------------------------------------------
def part6_what_turns_it_on():
    print("=" * 74)
    print("PART 6  the ratio only leaves 1 after an optimizer step")
    print("=" * 74)

    adv = advantages_from(REWARDS).detach()
    theta = torch.zeros(4, requires_grad=True)
    old_logps = NEW_LOGPS.detach()          # the frozen snapshot

    print(f"  {'inner epoch':>12} {'ratios':>34} {'# clipped @0.2':>15}")
    print("  " + "-" * 64)
    for epoch in range(1, 4):
        ratio = torch.exp((NEW_LOGPS + theta) - old_logps)
        cr = torch.clamp(ratio, 0.8, 1.2)
        obj = torch.minimum(ratio * adv, cr * adv)
        n = int((~torch.isclose(obj, ratio * adv)).sum())
        print(f"  {epoch:>12} "
              f"{str([round(r, 3) for r in ratio.tolist()]):>34} {n:>15}")

        # one optimizer step: theta moves, old_logps do not
        loss = -obj.mean()
        theta.grad = None
        loss.backward()
        with torch.no_grad():
            theta -= 5.0 * theta.grad        # exaggerated lr, to make it visible

    print("\n  -> epoch 1 always has ratio == 1 and clips nothing. Drift, and")
    print("     therefore clipping, only appears from epoch 2 onward.")
    print()
    print("     Epochs 2 and 3 show the SAME ratios, which is the brake")
    print("     working rather than a bug: torch.clamp has zero gradient")
    print("     outside its bounds, so once a rollout is clipped it stops")
    print("     contributing anything and cannot push the policy further.")
    print()
    print("     Practical rule: --clip_eps is what turns the clip on, and")
    print("     --inner_epochs is what gives it anything to clip. 7_4 ships")
    print("     inner_epochs=2 already, so lowering clip_eps alone is enough.\n")


def main():
    part1_ch06_loss()
    part2_the_ratio()
    part3_clipping()
    part4_eps_sweep()
    part5_onpolicy_is_identity()
    part6_what_turns_it_on()


if __name__ == "__main__":
    main()
