# SFT-08/RUN-03 b4 DEV preparation

This packet prepares a bounded, CPU-only replay of the frozen 75-row DAT-07 DEV evaluator panel through the banked b4 Q8 model. It does not start a model server, load model weights, or make a quality claim. Root owns the server process and the live run.

The client is [run03_b4_dev_client.py](run03_b4_dev_client.py). It checks every static identity before contacting the server, imports the pinned `run_eval.render_zeta2` implementation without a framework import, asks the server's native tokenizer for the prompt IDs, and sends native `/completion` requests. Every response is stored in the per-request JSONL together with the full prompt, target identity, native token IDs, request payload, raw completion response, and stop/cap metadata. JSONL rows are flushed and `fsync`ed individually; the summary is written with a file and directory `fsync`. An `fsync` error is fatal.

## Frozen inputs and matched legacy sources

| Item | Absolute path | Identity |
|---|---|---|
| DAT-07 DEV evaluator cases | `/mnt/e/sepalith/campaign-20260915/data-work/DAT-07-final-evaluator-cases.jsonl` | 75 rows, SHA256 `b16c5892635d6e9a5e9e789677057c85a5e875c907c163e91d39f25bc6ad3d21` |
| Banked b4 Q8 model | `/home/m0hawk/Documents/Sepalith/experiments/models/packaging_b4-Q8_0.gguf` | 2,012,011,904 bytes, SHA256 `e343feacbdb262f515c11c1b6b69c93781b3803f26184d810c5c3ed25f32512d` |
| Matched CPU server | `/home/m0hawk/Documents/Sepalith/experiments/bin/llama/llama-b10453/llama-server` | build 10453, commit `3cb7ffb1a`, SHA256 `123dc314b4a796bd091419171f5190966074460fa5863263847eb6b90208f804` |
| Legacy renderer | `/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb/experiments/eval/run_eval.py` | `run_eval.render_zeta2`, SHA256 `7fc6d4d796856ef3697365a462a8a1f5b0a9876d1f2e55cb88325a6ff7ef493d` |
| Scenario baseline reference | `/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb/experiments/eval/eval_scenarios.py` | SHA256 `da81802f553e91182ca7dbece30614ad5b12479f3f4e3f5a856a16dcd23fefc3` |
| No-op baseline reference | `/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb/experiments/eval/eval_noop_fp.py` | SHA256 `947d5dc6280bf85b831012116541f067f1104c46a682b3e3dc7ef722893f637d` |
| Native parser reference | `/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb/extensions/vscode-sepalith/src/extension.ts` | `parsePrediction`, SHA256 `368d6e502bb0f7c743ca5e38ec79c800c74b0776a14669bb1166985094217285` |

The client requires the exact case, model, server, renderer, baseline, and parser hashes. It also requires the server `/props` response to report the exact model path and `n_ctx=8192`. This prevents a same-name model or a smaller context from being silently used.

## Legacy projection and coverage

The PRM-03 `prompt_sha256` and prompt text are retained as source identity only. The b4 prompt is rendered afresh by `run_eval.render_zeta2`. The adapter consumes only `context.path`, `prefix`, `region_old`, `suffix_lines`, one optional `history.event_diff`, and the cursor's region line plus code-point/UTF-16 columns. `context.document_eol` is recorded but does not enter the legacy prompt.

The following PRM-03 fields are declared unsupported and never enter the prompt: `prompt_sha256`, `schema_version`, `replacement_range`, `diagnostics`, `retrieval`, `selected_references`, `scope_lines`, `scope_mode`, `source_ref`, and `source_provenance`. `region_new`, `target_body_text`, `operation`, and `family` are result labels; they are not prompt inputs. The client rejects operations other than `replace` and `no_op`. There are no delete rows in the frozen panel, and a delete fixture is rejected rather than mapped to an empty proposal.

Legacy zeta2 appends `<|user_cursor|>` at the end of a selected line. For a positioned cursor, the adapter verifies that the supplied UTF-16 column is the UTF-16 width of the code-point prefix, checks that the replacement preserves the bytes before the cursor, then losslessly splits the selected line at that cursor. The untouched line remainder and later old lines move into the legacy suffix. This gives the old renderer its line-end marker while keeping the full `region_new` array out of the prompt for scoring. An unpositioned cursor is accepted only for an empty region, which is the insertion representation used by the panel.

The 75-row projection has no exclusions and has 75 unique legacy prompt hashes:

| Family | Rows | Replace | No-op | Interior cursor split | Existing line-end cursor | Empty/unpositioned insertion |
|---|---:|---:|---:|---:|---:|---:|
| `finish_block` | 6 | 6 | 0 | 0 | 3 | 3 |
| `format_propagation` | 7 | 7 | 0 | 7 | 0 | 0 |
| `na_rm_propagation` | 6 | 6 | 0 | 6 | 0 | 0 |
| `no_op` | 32 | 0 | 32 | 3 | 26 | 3 |
| `pipe_rewrite` | 8 | 8 | 0 | 8 | 0 | 0 |
| `rename_propagation` | 8 | 8 | 0 | 8 | 0 | 0 |
| `roxygen_drafting` | 8 | 8 | 0 | 0 | 0 | 8 |
| **Total** | **75** | **43** | **32** | **32** | **29** | **14** |

Thus 61 rows have a positioned cursor and 14 are empty/unpositioned insertion rows. The replacement target is one line in 26 rows and multiple lines in 17 rows. The legacy 6,000-character advisory is exceeded by 29 adapted prompts; these rows are retained and must pass the server's native token/context check. The client refuses any prompt for which native prompt tokens plus the fixed production response cap would reach or exceed the server context.

No-op rows send an empty-proposal prompt. The `[NO_EDIT]` serialization is checked as an out-of-band label and is rejected if it leaks into the rendered prompt. Every row uses the extension's actual production request policy: the complete seven-stop list and `n_predict=320`, with `temperature=0`, `stream=false`, and `cache_prompt=false`. The native payload additionally requests `return_tokens=true` and `timings_per_token=true`. The historical scenario runner used only `[">>>>>>> UPDATED"]` with `n_predict=640`; those settings are recorded for comparison and deliberately never selected from a gold operation label. Operation and family affect scoring labels only.

## Denominators and durable artifacts

The summary reports denominators from successful native responses, so a transport or server failure cannot become a score. On a complete 75-row run the expected denominators are:

| Measure | Denominator | Per-row rule |
|---|---:|---|
| Exact replacement | 43 | `renderer.norm(parsed_prediction) == renderer.norm(region_new)` |
| No-op correctness | 32 | no parsed proposal is correct; any parsed line is a false suggestion |
| Parser | 75 | successful response passed the pinned `parsePrediction` path; parser output is retained even when empty |
| Generation cap | 75 | successful response; cap hit is native `stop_type == "limit"` or `tokens_predicted >= n_predict` |

The fresh ext4 output directory is `/home/m0hawk/.local/state/sepalith-campaign-20260915/SFT-08-b4-dev-v1`. The client creates and refuses to overwrite:

- `/home/m0hawk/.local/state/sepalith-campaign-20260915/SFT-08-b4-dev-v1/per_request.jsonl`
- `/home/m0hawk/.local/state/sepalith-campaign-20260915/SFT-08-b4-dev-v1/summary.json`

The source cases live on `/mnt/e` (9p). Outputs stay under `/home` ext4 because the `/mnt/h` directory `fsync` path has returned `EIO` in this campaign. After a complete run, root may create `/home/m0hawk/.local/state/sepalith-campaign-20260915/SFT-08-b4-dev-v1.tar` from the two files and record all three SHA256 values. The client does not overwrite an existing output, summary, or temporary summary file.

Each transport or model error is retained as a per-request row with its error text, excluded from score denominators, and changes the run status to `failed` with a nonzero exit code. A soft deadline returns status `deadline` and exit code 124.

## Root launch and bounded client command

Root should verify the port owner before starting the server. The pinned server command is:

```sh
set -euo pipefail
BIN=/home/m0hawk/Documents/Sepalith/experiments/bin/llama/llama-b10453/llama-server
MODEL=/home/m0hawk/Documents/Sepalith/experiments/models/packaging_b4-Q8_0.gguf
test "$(sha256sum "$BIN" | cut -d ' ' -f1)" = 123dc314b4a796bd091419171f5190966074460fa5863263847eb6b90208f804
test "$(stat -c %s "$MODEL")" = 2012011904
test "$(sha256sum "$MODEL" | cut -d ' ' -f1)" = e343feacbdb262f515c11c1b6b69c93781b3803f26184d810c5c3ed25f32512d
exec env CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  "$BIN" -m "$MODEL" --alias sepalith --temp 0 --host 127.0.0.1 --port 18099 \
  -c 8192 --parallel 1 -t 2 -tb 2 -ngl 0
```

With that root-owned server answering on loopback, the bounded benchmark command is:

```sh
set -euo pipefail
RUN_DIR=/home/m0hawk/.local/state/sepalith-campaign-20260915/SFT-08-b4-dev-v1
mkdir -p "$RUN_DIR"
test ! -e "$RUN_DIR/per_request.jsonl"
test ! -e "$RUN_DIR/summary.json"
export CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export MKL_NUM_THREADS=2
export SEPALITH_B4_DEV_SOFT_DEADLINE_S=840
export SEPALITH_B4_DEV_HARD_DEADLINE_S=900
export SEPALITH_B4_DEV_CHECKPOINT_RESERVE_S=60
/usr/bin/timeout --signal=TERM --kill-after=30s 900s \
  /usr/bin/python3 docs/campaign/work/b4-dev/run03_b4_dev_client.py \
  --url http://127.0.0.1:18099 \
  --cases /mnt/e/sepalith/campaign-20260915/data-work/DAT-07-final-evaluator-cases.jsonl \
  --model /home/m0hawk/Documents/Sepalith/experiments/models/packaging_b4-Q8_0.gguf \
  --server-binary /home/m0hawk/Documents/Sepalith/experiments/bin/llama/llama-b10453/llama-server \
  --repo-root /home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb \
  --output "$RUN_DIR/per_request.jsonl" \
  --summary "$RUN_DIR/summary.json"
```

The client soft deadline is 840 seconds, leaving the 60-second checkpoint/output reserve inside the 900-second hard deadline. A completed row is durable before the next request begins. A native response retains `content`, token IDs, timing data, `stop`, `stop_type`, `stopping_word`, `tokens_predicted`, `tokens_evaluated`, `tokens_cached`, `truncated`, `has_new_line`, and the full server JSON. The client never terminates the server.

After the client reports `status=complete`, root can archive the fresh files:

```sh
set -euo pipefail
RUN_DIR=/home/m0hawk/.local/state/sepalith-campaign-20260915/SFT-08-b4-dev-v1
ARCHIVE=/home/m0hawk/.local/state/sepalith-campaign-20260915/SFT-08-b4-dev-v1.tar
test "$(/usr/bin/python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["status"])' "$RUN_DIR/summary.json")" = complete
tar --sort=name --mtime='UTC 1970-01-01' --owner=0 --group=0 --numeric-owner \
  -cf "$ARCHIVE" -C "$RUN_DIR" per_request.jsonl summary.json
sha256sum "$RUN_DIR/per_request.jsonl" "$RUN_DIR/summary.json" "$ARCHIVE"
```

The client and tests are framework-free. The focused test command is:

```sh
PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 \
  docs/campaign/work/b4-dev/test_run03_b4_dev_client.py
```

It exercises the actual 75-row projection and renderer, prompt identity isolation, no-op and delete behavior, UTF-16/code-point geometry, native request fields and stop metadata, parser scoring, and deadline reserve validation. One fixture mutates operation/family/target/source labels and confirms the prompt settings remain fixed. The test suite completed with 8 passing tests in under 0.1 seconds during preparation. No CUDA device, model load, server, network request, or cloud job was used.

Live gates remain with root: port ownership and server startup, native tokenization of all 75 adapted prompts, complete response/archive hashes, and the resulting b4 quality readout. The historical b4 comparator is preserved as context; this preparation does not upgrade it to evidence for the current run.
