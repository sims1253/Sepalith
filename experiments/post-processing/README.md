# Post-processing

Cleans, licenses, tags, and assembles datasets after mining or generation.

| Script | What it does |
|---|---|
| `normalize_external.py` | Runs `air format` + `jarl check --fix` on R code inside harvested and synthetic records. Keeps both the original and the normalized text. Unparseable blocks stay, tagged with a reason. |
| `enrich_provenance.py` | Adds `source_url` and `license` to every record. Joins CRAN provenance for corpus-derived sets, parses repo licenses for mined sets, reconstructs full prompts for synthetic sets. Templates come from the generator sources by AST, so they cannot drift. |
| `style_tag.py` | Tags each record tidyverse, base, or neutral. A TOSEM study found mixed-style training hurts; the tag lets mixtures stratify. |
| `pull_licenses.py` | Extracts each package's LICENSE file from its tarball into provenance. |
| `push_hf.py` | Pushes shards, manifest, and provenance to the private HF dataset repo. |
| `estimate_tokens.py` | Token-budget estimate for pretraining: stratified sample, exact tokenizer counts, per-area ratios. |
| `format_sft_v1.py` | Renders finish-block records into the Zeta-2 prompt format. |
| `format_sft_types.py` | Builds the type-conditioning ablation pair: same records, with and without a `<filename>types` section from `ry dump-types`. |
| `assemble_sft_v2.py` | Mixes all families into train/eval sets. `--out` to target a directory; skips missing inputs with a note. |
| `datatrove_dedup.py` | WIP: exact-seq plus minhash dedup over the corpus. |
