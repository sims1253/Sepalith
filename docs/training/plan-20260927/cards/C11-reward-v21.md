# C11. Reward v2.1 and RL length policy

- **Where:** PC, daytime, CPU only (needs R 4.6.1 and `air`, both installed)
- **Needs:** C01
- **Produces:** reward v2.1 with tests, and the RL data code on the approved
  length policy
- **Effort:** one to two agent-days

## Goal

The census (C12), RFT and DPO (C13), and any GRPO (C15) all score samples with
this reward. Reward v2 has a known parser bug. It also scores only exact
matches, so a valid alternative edit counts as wrong, and binary rewards give
little signal.

## Read first

- AUDIT.md, section RL preparation
- `docs/campaign/receipts/RL-11-reward-v2-root-parser-review.json`,
  `RL-11-reward-review.json` and `RL-11-length-policy-root-review.json`
- The reward and RL data modules from C01: `campaign_reward_v2`,
  `campaign_r_parse_probe`, `reward_parse_only.R` and `campaign_rl_data`

## Steps

1. **Parser fix.** Treat only the designated syntax-error exit code as invalid
   R. Any other nonzero exit, signal or timeout raises an infrastructure error
   that is logged and retried, never scored as a model failure. Record the R
   version the probe used.
2. **Canonical comparison.** Canonicalize both candidate and reference
   regions: run `air format` on the reconstructed file region, normalize line
   endings and strip trailing whitespace. `exact` means canonical equality. If
   `air` cannot format either side, fall back to the raw comparison and flag
   the case.
3. **Graded reward.** Keep exact match dominant.

   | Outcome | Reward |
   | --- | --- |
   | Correct edit (canonical `exact`) | 1.0 |
   | Edit that parses but is not exact | 0.5 × a canonical line-level similarity in [0, 1] |
   | Edit that does not parse | 0 |
   | No-op case, model answers `[NO_EDIT]` | 1.0 |
   | No-op case, model suggests any edit | −0.5 |
   | Protocol-invalid output or cap hit | −0.5 |

   The partial credit can never exceed 0.5, so a near miss always ranks below
   a correct edit.

   Keep `exact` as a separate field. Rules and pass rates use `exact` only,
   never the graded value.
4. **Tests.** Include fixtures for:
   - alternative valid formatting of the same edit (exact after
     canonicalization);
   - a semantically different edit (partial credit);
   - a parse failure;
   - a no-op true positive and false positive;
   - an infrastructure failure, such as a killed R process, which must raise;
   - a protocol-invalid output.
5. **Length policy.** Replace the hardcoded 2,048 prompt and 192 completion
   limits with the reviewed policy: 3,072 prompt tokens including BOS, 1,024
   completion tokens including EOS, 4,096 context. Show that it accepts all
   15,006 TRAIN rows, as the review found.
6. Update RL-12's stale task text (Monday cutoff, theta0 and b4 identities) in
   the campaign task list snapshot with a pointer to this plan. Alternatively,
   add a note in `docs/campaign/` that the plan supersedes it.

## Acceptance

- The tests pass.
- Scoring the 15,006 TRAIN targets against themselves gives `exact` = 1 for
  every row. Report any exception as a canonicalization bug.
- The throughput is recorded in `status/C11.json`: seconds per 1,000 samples on
  the PC with 8 threads. C12 needs this to size CPU scoring.
