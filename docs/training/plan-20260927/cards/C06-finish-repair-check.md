# C06. Spot-check the repaired finish_block rows

- **Where:** PC, daytime, CPU only
- **Needs:** nothing
- **Produces:** a defect-rate receipt for the 3,503 repaired finish_block rows
- **Effort:** half an agent-day

## Goal

finish_block makes up 52% of the SFT data, and 3,503 of its 7,785 rows went
through a repair pipeline. An earlier SFT round (SFT-06) failed partly on
finish quality. Before the first night, confirm that repaired targets are
sound.

## Read first

- `docs/campaign/work/corrective-sft-postmortem/postmortem.md` and
  `docs/campaign/work/corrective-finish-quality-review/`
- `docs/campaign/receipts/DAT-10-train-finish-source-repair-v3-root-review.json`
- The repair ledger at
  `/mnt/e/sepalith/campaign-20260915/data-work/DAT10-finish-source-repair-v3/repair-ledger.jsonl`

## Steps

1. Draw a stratified random sample of 80 repaired rows, spread across repair
   types in the ledger, plus 20 unrepaired finish_block rows as a control.
   Use seed 3407.
2. For each row, decode the target with the parent tokenizer. Then check that:
   - the target parses with R (`Rscript -e 'parse(text=...)'` on the
     reconstructed region);
   - `air format` is stable on it;
   - the edit completes the block that the prompt opens, without swallowing
     code after the cursor;
   - braces and indentation are consistent with the surrounding file.
3. Classify each row as correct, cosmetic issue or defective, with a
   one-line reason. Use the postmortem's categories where they fit.
4. Write the receipt with counts, the defective examples (as ids plus a short
   description), and the defect rate with a 95% interval.

## Acceptance and rule

- A defect rate of at most 5% in the repaired sample, and no worse than the
  control plus 3 points, means continue.
- A rate between 5% and 10% means continue, but list the defective repair
  types so C04 can drop those types if the user agrees.
- A rate above 10% means stop-for-user, before any SFT night.
