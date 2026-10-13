# RL-12 is superseded

Task RL-12 in [`tasks.json`](tasks.json) ("Conditionally improve the second
parent with diagnosed RL") is stale. This snapshot is frozen, so the task text
stays as it was; this note records what replaced it.

- The Monday stop and cutoff times no longer apply. Post-SFT work now runs in
  the PC's nightly window (01:00 to 09:00 Europe/Berlin) and on free Kaggle
  GPUs.
- The "theta0 Q8 and b4" identities are no longer the protected parents. The
  editing-SFT parent is CPT checkpoint 11,586, and later parents are chosen by
  the pre-registered rules.
- Post-training goes offline first (RFT, then DPO). Online GRPO runs only if
  rule R4 finds enough signal.

The [post-CPT plan](../training/plan-20260927/README.md) replaces RL-12. See its
cards C11 (reward v2.1 and RL length policy), C12 (pass-rate census), C13 (RFT
and DPO), C14 (GRPO gate) and C15 (GRPO pilot), and the decision rules in
[RULES.md](../training/plan-20260927/RULES.md).
