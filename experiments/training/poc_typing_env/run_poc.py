#!/usr/bin/env python3
"""Run the typing-RL PoC loop (see env.py docstring for the reward design).

The visible demonstration of the RL loop without a GPU trainer:

  for round r, task t:
    K rollouts: the glm-5.3 typer streams a solution; the env scores the
    typing (process, decision tier) and the artifact (execution, benchmark,
    judge cascade);
    best rollout by scalar_total becomes the FEEDBACK context for round
    r+1: its code + component breakdown + mechanically derived directives
    ("tests basic/na_window failed", "1.4x slower than baseline", ...).

That is reward-conditioned iterative refinement — the same loop GRPO closes
with gradients over a policy; here the "policy update" is the feedback
prompt, so the reward machinery, cascade and logs are exercised end-to-end.

Usage (ZAI_API_KEY in env; d3 optional — TYPING_ENV_D3=auto|off|stub):
  python3 run_poc.py --tasks all --rounds 3 --rollouts 3
  python3 run_poc.py --tasks rolling_mean --rounds 1 --rollouts 1 --quick
  python3 run_poc.py --glm-off          # stub everything, no network

Writes (gitignored): results.jsonl; and RESULTS.md + dashboard.html
(these two are committed — numbers belong to the repo per poc convention).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from d3_client import make_judge
from env import EpisodeRow, EnvConfig, TypingRLEnv
from glm_client import GlmClient
from policy import GlmTypist
from tasks import TASKS
from reports import write_dashboard, write_results_md

HERE = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# feedback construction (mechanical, from the reward breakdown)
# ---------------------------------------------------------------------------

def build_feedback(row: EpisodeRow, best_code: str) -> str:
    parts = [f"Best solution so far (scalar {row.scalar_total}, "
             f"correctness {row.correctness:.2f}):\n```r\n{best_code}\n```"]
    if row.test_rows:
        fails = [t for t in row.test_rows if not t["pass"]]
        if fails:
            why = "; ".join(f"{t['name']}: {t['why'][:60]}" for t in fails)
            parts.append(f"FAILING tests: {why}")
        else:
            parts.append("All tests pass.")
    elif row.test_error:
        parts.append(f"Code failed to run: {row.test_error}")
    if row.base_s and row.bench_s:
        parts.append(f"Benchmark: {row.base_s / row.bench_s:.1f}x the naive "
                     f"baseline ({row.bench_s:.3f}s vs {row.base_s:.3f}s) — "
                     "vectorize harder if below 4x.")
    else:
        parts.append("Benchmark failed or errored — must handle the "
                     "benchmark data.")
    j = row.glm_judge
    if j and not j.get("parse_error"):
        parts.append(f"Judge notes: {j.get('notes', '')} "
                     f"(elegance {j.get('elegance')}, r_idiom "
                     f"{j.get('r_idiom')}, reward_hacking "
                     f"{j.get('reward_hacking')}).")
    d3q = row.d3_terminal.get("quality", {}).get("label")
    if d3q:
        parts.append(f"Decision-model quality: {d3q}.")
    parts.append("Rewrite the function to fix every issue above. Same "
                 "output contract, one ```r block only.")
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# one rollout
# ---------------------------------------------------------------------------

def run_rollout(task, typer, judge, glm, feedback, temperature
                ) -> tuple[EpisodeRow, str]:
    env = TypingRLEnv(task, judge=judge, glm=glm, cfg=EnvConfig(
        max_chars=task.max_chars))
    env.reset()
    gen = typer.type_solution(task.prompt, feedback=feedback,
                              temperature=temperature)
    full = ""
    stream_error = ""
    try:
        while True:
            chunk = next(gen)
            _, _r, done, _info = env.step(chunk)
            if done:
                break
        full = env.prefix
    except StopIteration as st:                       # stream ended first
        full = st.value or env.prefix
    except Exception as e:                            # noqa: BLE001
        # mid-stream death: the typed-so-far prefix is the episode the
        # env saw — finalize it honestly (no silent regeneration)
        stream_error = f"{type(e).__name__}: {str(e)[:120]}"
        full = env.prefix
        if not env.done:
            env.step("\n```")                         # close the episode
    row = env.finalize()
    if stream_error:
        row.test_error = (row.test_error + " | " if row.test_error else "") \
            + f"stream_died: {stream_error}"
    return row, full


# ---------------------------------------------------------------------------
# main loop
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tasks", default="all",
                    help="comma list or 'all'")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--rollouts", type=int, default=3)
    ap.add_argument("--temp-first", type=float, default=0.2,
                    help="round-0 temperature (near-greedy baseline)")
    ap.add_argument("--temp", type=float, default=0.8,
                    help="exploration temperature, rounds > 0")
    ap.add_argument("--d3", default=None,
                    help="auto|d3|d3:<model_id>|stub|off")
    ap.add_argument("--glm-off", action="store_true",
                    help="no network: mock typer, no glm judge")
    ap.add_argument("--out", default=str(HERE / "results.jsonl"))
    ap.add_argument("--results-md", default=str(HERE / "RESULTS.md"))
    ap.add_argument("--dashboard", default=str(HERE / "dashboard.html"))
    ap.add_argument("--quick", action="store_true",
                    help="1 task, 1 round, 1 rollout — end-to-end smoke")
    ap.add_argument("--no-resume", action="store_true",
                    help="start fresh instead of skipping rollouts "
                         "already present in --out")
    args = ap.parse_args()

    if args.quick:
        args.tasks, args.rounds, args.rollouts = "rolling_mean", 1, 1

    task_ids = (list(TASKS) if args.tasks == "all"
                else [t.strip() for t in args.tasks.split(",")])
    tasks = [TASKS[t] for t in task_ids]

    judge = None
    if not args.glm_off:
        try:
            judge = make_judge(args.d3 or "auto")
        except Exception as e:                        # noqa: BLE001
            print(f"[warn] decision model unavailable ({e}); "
                  "continuing without it")
    glm = None if args.glm_off else GlmClient()

    rows: list[dict] = []
    # resume: rows already in --out are kept; their (task, round, rollout)
    # slots are skipped (crash resilience — rows are appended + flushed
    # after every rollout)
    done_keys: set[tuple] = set()
    if not args.no_resume and Path(args.out).exists():
        for line in open(args.out):
            try:
                r = json.loads(line)
            except ValueError:
                continue
            rows.append(r)
            done_keys.add((r["task_id"], r.get("round"),
                           r.get("rollout")))
        if done_keys:
            print(f"[resume] {len(done_keys)} rollouts already in "
                  f"{args.out}", flush=True)
    out_fh = open(args.out, "a")

    def best_so_far(task_id: str, before_round: int) -> tuple | None:
        """Round-winner selection on the PRIVATE score first (wiki
        aide2-rsi: survival decided on a score the agent never sees):
        held-out correctness, then the full scalar."""
        cands = [r for r in rows if r["task_id"] == task_id
                 and r.get("round", 0) < before_round]
        if not cands:
            return None
        return max(cands, key=lambda r: (
            r.get("heldout_correctness")
            if r.get("heldout_correctness") is not None else -1.0,
            r.get("scalar_total", -9)))

    t_start = time.time()
    for task in tasks:
        best_code = ""
        for rnd in range(args.rounds):
            prev = best_so_far(task.task_id, rnd)
            feedback = None
            if prev is not None:
                feedback = build_feedback(
                    EpisodeRow(**{k: v for k, v in prev.items()
                                  if k in EpisodeRow.__dataclass_fields__}),
                    prev.get("code", ""))
            for k in range(args.rollouts):
                if (task.task_id, rnd, k) in done_keys:
                    continue
                temp = args.temp_first if rnd == 0 else args.temp
                t0 = time.time()
                if args.glm_off:
                    from policy import MockTypist
                    typer = MockTypist(
                        "```r\nf <- function(x) x\n```")
                else:
                    typer = GlmTypist(glm, temperature=temp)
                row, _full = run_rollout(task, typer, judge, glm,
                                         feedback, temp)
                row.round, row.rollout = rnd, k          # type: ignore[attr-defined]
                row.temperature = temp                   # type: ignore[attr-defined]
                dt = round(time.time() - t0, 1)
                rows.append(json.loads(json.dumps(row.__dict__)))
                out_fh.write(json.dumps(rows[-1], ensure_ascii=False)
                             + "\n")
                out_fh.flush()
                print(f"[{task.task_id}] r{rnd} k{k} "
                      f"scalar={row.scalar_total:+.3f} "
                      f"(c={row.correctness:.2f} "
                      f"ho={row.heldout_correctness if row.heldout_correctness is None else round(row.heldout_correctness, 2)} "
                      f"gap={row.rh_gap} "
                      f"p={row.perf} "
                      f"s={None if row.style is None else round(row.style, 2)} "
                      f"g={None if row.guard is None else round(row.guard, 2)}) "
                      f"proc={row.process_reward:+.3f} "
                      f"route={row.skip_reason or 'run'} {dt}s",
                      flush=True)
    out_fh.close()

    with open(args.out, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    usage = glm.usage if glm else {"prompt_tokens": 0,
                                   "completion_tokens": 0, "calls": 0}
    meta = {
        "rounds": args.rounds, "rollouts": args.rollouts,
        "glm_off": args.glm_off,
        "d3_backend": getattr(judge, "backend", "off"),
        "wall_s": round(time.time() - t_start, 1),
        "glm_usage": usage,
        "generated_at": time.strftime("%Y-%m-%d %H:%M"),
    }
    write_results_md(rows, meta, Path(args.results_md))
    write_dashboard(rows, meta, Path(args.dashboard))
    print(json.dumps(meta, indent=1))
    print(f"wrote {args.out}, {args.results_md}, {args.dashboard}")


if __name__ == "__main__":
    main()
