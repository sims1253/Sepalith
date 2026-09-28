# Full-corpus CPT trainer v2 independent review

This packet reviews the frozen `r2-full-weight-cpt-full-corpus-trainer-v2` closure without modifying it or reading model tensor payloads.

The source and artifact manifests reproduce exactly, and the nine packet tests pass independently. The packet is still blocked for binding: `root-admission.template.json` contains recipe hash `fac304a8...`, while the actual supplied `recipe.template.json` hash is `09349f90...`. The binder rejects that mismatch at its first identity gate.

The prepared CPT and full-weight editing SFT trainers both explicitly allow only 2K, 4K, 8K, 16K, or 32K contexts. Their validators and collators receive the selected limit and reject oversized rows; they do not truncate them. The accepted CPT rechunker separately emits only 8K, 16K, and 32K profiles. These are preparation and admission constraints. The model configuration advertises 131072 positions, but that metadata is not a 64K or 128K runtime-fit result.

Run the bounded reproduction with:

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 python3 verify_review.py
```
