# PRM-01 worker return

TASK: PRM-01  
STATUS: verified  
OWNER: worker-prompt  
STARTED: 2026-09-12T03:17:45+02:00  
ENDED: 2026-09-12T03:24:38+02:00  

DEPENDENCIES_CHECKED: PRE-01 (`docs/campaign/receipts/PRE-01-source-snapshot.json`, accepted, snapshot `20260912T005156Z`); PRE-05 (`docs/campaign/receipts/PRE-05-environment-lock.json`, accepted). The package records the plan, canonical, owner and execution source roles separately. The canonical extension snapshot used for behavioral basis is `07855573d974e76582acef99c7642b26dd8b3e4b6cbbd96126ad59107fe1cec8`; owner/execution source remains read-only.

ACTION: Built `docs/campaign/work/prompt/build_event_fixtures.py` and generated `docs/campaign/work/prompt/event-fixtures.json`. The package has 11 synthetic functional fixtures with stable IDs `PRM01-E01` through `PRM01-E11`. It covers typing, deletion, cursor move, history append and eviction, diagnostic refresh, referenced-definition change, truncation-anchor move, file switch and no-op. It includes long-scope, cross-file, local-only, missing-evidence and no-op strata.

The input constructor takes pre-edit lines, bounded history and evidence only. It has no target parameter. Targets are separate out-of-band oracles with replacement geometry, cursor geometry, identity and output terminator. Each file identity is computed from the captured pre-edit UTF-8 content. History records retained order, capacity, append and eviction IDs. Missing providers are represented by empty evidence and explicit missing provider states.

The no-op boundary fixture preserves quoted R source containing `<[fim-middle]>`, `</s>` and `<|endoftext|>` plus trailing spaces. It records pinned-tokenizer probes for `add_special_tokens=False` and `True`, including BOS `0` and EOS `1` metadata. Quoted marker-like source is admitted and preserved; an ambiguous full delimiter line is marked reject-until-the-output-parser-contract-is-frozen. No new prompt vocabulary or provider evidence was added.

COMMANDS_OR_METHOD:

```text
python3 docs/campaign/campaign.py brief PRM-01
/home/m0hawk/Documents/Sepalith/.venv-sft/bin/python docs/campaign/work/prompt/build_event_fixtures.py --write
/home/m0hawk/Documents/Sepalith/.venv-sft/bin/python docs/campaign/work/prompt/build_event_fixtures.py --check
PYTHONPATH=packages/sepalith/src /home/m0hawk/Documents/Sepalith/.venv-sft/bin/python -m unittest discover -s packages/sepalith/tests -p 'test_protocol.py'
```

RESULT: `--write` and `--check` both pass. Protocol tests pass: 9 tests. Fixture artifact SHA256 is `901c076b61fcd06194aae10b94edc9df900e521d57a7eec075577be2bd21a944`; builder SHA256 is `7208688e5b340a00c334803e6fb36716a40b55e20f988f916b6d86333c04364f`. Renderer is `zeta2-v1`. Tokenizer is MiniCPM Midtrain revision `8dc5f6055b90fe4b9422340810b270b9569f37f3`, tokenizer JSON SHA256 `3e065a558a034185fe299917b398685c1facd0169a9eea1e629eb30c171fed81`, measured with `add_special_tokens=False`. Long prompt counts are 372 tokens (`PRM01-E05`) and 507 (`PRM01-E08`).

Coverage counts: 2 typing, and 1 each of the other nine named event classes; 2 long-scope, 2 cross-file, 5 local-only, 1 missing-evidence and 1 no-op. Observed real event fixtures: 0. Synthetic functional fixtures: 11.

ACCEPTANCE: pass — all named classes have stable IDs and source/package identity; pre-edit input, targets and future intent are separated; exact whitespace and Unicode code-point/UTF-16 cursor geometry are retained; history append/eviction, missing evidence, marker boundaries, renderer identity and tokenizer identity are executable checks; no final-test rows are included.

CHANGED_FILES:

- `docs/campaign/work/prompt/build_event_fixtures.py`
- `docs/campaign/work/prompt/event-fixtures.json`
- `docs/campaign/receipts/PRM-01-event-fixtures.json`
- `docs/campaign/receipts/PRM-01-worker.md`

ARTIFACTS: `docs/campaign/receipts/PRM-01-event-fixtures.json`; `docs/campaign/work/prompt/event-fixtures.json`; executable builder and validation command above.

UNRESOLVED: No approved real editor event-log or document-change trace was located in the bounded audited paths for this packet. Git/commit examples and synthetic scenario rows were excluded because they contain post-edit or target state. The bounded audit does not establish absence outside those paths. Synthetic rows must remain labelled functional tests and must not be counted as observed coverage.

NEXT: Lead reviews and accepts PRM-01, then assigns the renderer/stop-boundary gap card using this package while preserving the canonical-versus-owner source distinction.

LEASE_RELEASED: yes — CPU-only work completed with at most two threads; no CUDA import/context, server, cloud/API generation, dataset mutation or final-test access.
