# CPT prefix extension v1

Status: prepared and unlaunched. The combined destination corpus/cache is not
yet bound or admitted.

This source resumes the verified full-state checkpoint at global step 82 while
retaining the original optimizer/scheduler/RNG stage offset 66. Its sampler
starts the extended schedule at cursor 256. The destination schedule must keep
all 183,084 old unique row IDs in their exact order before new unique rows. The
four old alignment replays are after the unconsumed unique prefix and may be
replaced by the combined schedule's final alignment tail.

`prefix_extension_contract.py` performs the expensive one-time binder check:

- hashes every old and new cache payload against its immutable manifest;
- requires the complete old input/label binaries to be byte prefixes of the
  destination binaries;
- compares every old row index and document provenance record;
- streams the destination source JSONL and reconstructs input/label hashes,
  row hashes, token counts, loss counts, BOS/EOS boundaries and package/document
  totals;
- joins every destination schedule ID to its stored draw ordinal;
- checks checkpoint cursor 256, source step 82, offset 66, and a later stop.

The destination cache builder must therefore append physical rows to the old
cache layout. A cache that merely contains equivalent rows in another physical
order is deliberately rejected.

Root binding sequence after the combined cache is terminal:

1. Fill `root-admission.template.json` from the reviewed combined data receipt
   and `source-checkpoint82.preparation.json`.
2. Set `mandatory_stop_stage_step` to an admitted stage-local value greater
   than 16; cadence is measured from offset 66.
3. Run `root-commands.json.bind`, then its CPU preflight. Both execute the full
   prefix/conservation check.
4. Only root may create launch admission and run the guard command.

All 20 CPU tests pass, including an actual Transformers Trainer test where a
checkpoint after an old consumed prefix resumes on an extended dataset at the
first unseen row and matches uninterrupted model, optimizer, and scheduler
state.
