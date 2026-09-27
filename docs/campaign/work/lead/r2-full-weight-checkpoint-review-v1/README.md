# Full-weight checkpoint independent review

This packet reviews the frozen `r2-full-weight-checkpoint-v1` helper and replays
the resulting regressions against the superseding `r2-full-weight-checkpoint-v2`.
It does not modify either source packet and does not load tensor payloads onto a
model or accelerator.

The v1 review found a blocking missing kind boundary: `require_full=True`
accepted a resumable adapter checkpoint, and the manifest kind was not bound to
the inventoried campaign state. V2 adds `expected_checkpoint_kind`, binds the
manifest to campaign state, and revalidates the dense layout at resume. The six
independent v2 regressions and the twelve producer tests pass.

Long-run integration remains conditional on the training entrypoint calling
`verify_checkpoint(..., require_full=True,
expected_checkpoint_kind="full_weights")`, proving a real optimizer/RNG resume,
and budgeting the callback's full-weight output cadence. A truthy sampler object
is presence validation; its schema and restoration semantics remain the
trainer's responsibility.

Run with:

```sh
env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  MKL_NUM_THREADS=2 nice -n 10 python3 -B -m unittest -v \
  test_v2_checkpoint_contract.py
```
