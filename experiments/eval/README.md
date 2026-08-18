# Eval

Measures edit quality and latency. The core lesson from Stage 0b: score the
completion, not the whole region, and always report the copy-from-context
baseline — a model that predicts "no change" otherwise looks good.

| Script | What it does |
|---|---|
| `build_examples.py` | Builds held-out next-edit examples from commit diffs of cloned repos. Commit dates after 2026-04-15 keep them out of every public model's training data. |
| `run_eval.py` | Renders prompts in the official Zeta-1/2/2.1 formats and scores predictions. `--variant midtyping` puts the cursor mid-line on a partial prefix of the first changed line, where the copy baseline scores 0. `--align suffix` realigns whole-region outputs for fair scoring. Stores raw predictions so rescoring needs no re-inference. |
| `analyze.py` | Aggregates results with copy-baseline and bootstrap CIs. |
| `audit.py` | Auto-flags mined examples for human review. |
| `keystroke_sim.py` | Measures keystroke-to-suggestion latency against a llama-server: cold start vs warm prefix cache, real R context. |
| `run_sims.sh` | Runs the simulator across context sizes with a fresh server per size, so cold numbers stay honest. |

Latency numbers from this machine are pessimistic: it shares duty with other
work. Re-bench on a quiet box before citing.
