# C04. Editing-SFT recipe v2: schedule, gates, deadline stop

- **Where:** PC, daytime, CPU only
- **Needs:** C01
- **Produces:** a bound, tested SFT v2 recipe that nights C07 to C09 can run
- **Effort:** one to two agent-days

## Goal

Fix the problems the audit found in the prepared SFT packet, then pre-bind
everything the nights need. Arms must run unattended, stop at the deadline,
and resume the next night.

## Read first

- AUDIT.md, section Editing SFT packet
- RULES.md, rules R1 and R2
- The consolidated SFT code from C01, and the original packet
  `docs/campaign/work/lead/r2-full-weight-edit-sft-preparation-v1/`

## Steps

1. **Schedule v2.** Build a new draw schedule over the same 15,006 rows. Keep
   the effective batch at 16 (micro-batch 1, gradient accumulation 16) with a
   fixed composition of 13 edit plus 3 no-op per batch.
   - Each edit row appears once. Pad the last batch with at most 11 edit
     replays.
   - Spread no-op rows evenly, with at most 3 exposures per row: 1,025 rows
     seen 3 times and 69 seen twice.
   - Expect 1,071 updates, an 18.75% no-op share, and gates at updates 268,
     536, 803 and 1,071.

   Use seed 3407 and draw no-op replays only after each pool is exhausted.
   Emit the same `coverage` fields the old schedule had, and test them.
2. **Optimizer and schedule defaults** (pre-registered here):
   - aurora_mix with the same mechanics as CPT; hidden LR equals the arm's
     LR and side LR is LR / 10
   - warmup of 32 updates, then cosine decay to 10% of peak at update 1,071
   - weight decay 0

   Arms differ only in LR. Keep the binder's upper bound of 3e-5, and document
   why: it is 10 times the CPT hidden LR.
3. **Pilot support.** Let one parent start several arms that share the
   schedule, each stopping at the first gate. The winning arm then resumes
   from its own gate checkpoint with its exact state: optimizer, scheduler,
   RNG and sampler cursor.
4. **Automatic gates.** At each gate:
   - save full state;
   - run DEV75 and DEV250 generation in a separate process that frees the
     trainer's GPU memory (the save, stop, evaluate, resume pattern);
   - apply R1;
   - write the decision record that `milestone_gate.py` expects, with
     `continue_training` set from the rule and the rule inputs attached;
   - continue, stop-for-user or roll back.

   A human override file, when present, takes precedence. Keep every existing
   fail-closed binding: panel hash, recipe hash, checkpoint manifest.
5. **Deadline stop.** Honour `SEPALITH_DEADLINE_EPOCH` from C02. After the
   deadline, finish the current optimizer update, save full state and exit
   with a resumable status. The next night resumes exactly. If a gate
   evaluation would not finish before the deadline, save and defer it to the
   next night.
6. **Pre-update masking gate.** Port the check from
   `docs/campaign/work/corrective-target-only-gates-v1/`. Before update 1,
   verify on the live trainer's first batch that prompt tokens are masked and
   that body, terminal and EOS are supervised. Abort otherwise.
7. **Template hygiene.** Label example values in admission templates as
   `EXAMPLE_ONLY`, so nobody copies them as decisions.
8. **Bind the parent.** Use checkpoint 11,586 with the hashes in CONTEXT.md.
   Produce the bound recipe for three pilot arms at LR 3e-6, 1e-5 and 3e-5,
   and the job files C07 and C08 will enqueue. Do not enqueue them yet.
9. **Tests.** Cover schedule coverage and exposure caps, the gate rule on
   synthetic metric inputs for all three outcomes, deadline save and resume on
   a tiny CPU model, the masking gate, and override precedence.

## Acceptance

- The tests pass.
- The schedule histogram shows edits at exactly 1, apart from at most 11
  replays, and no-ops at no more than 3.
- The bound recipes and job files exist and are hash-listed in `status/C04.json`.

## Stop and ask if

- The existing trainer cannot resume mid-schedule with exact state.
- The gate evaluation cannot run in a separate process on 32 GB after the
  trainer frees its memory.
