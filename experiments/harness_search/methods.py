"""methods.py — the H1 bake-off arms (plan §2) + the resumable driver.

Arms (budget-matched, ~13 iterations x M=3 = 39 evaluated candidates each,
plus the SHARED default-config baseline evaluation common to all arms):

  hill       single-lineage greedy hill-climb (pi-autoresearch pattern):
             keep best-so-far, propose 3 mutations, take the winner if it
             clears the guardrails, revert on regression.
  population population evolution (ShinkaEvolve pattern): archive of every
             evaluated config, proposer may re-mix any two archived parents;
             2 demes, cross-deme parent choice at iterations 7/10/13.
  gepa       prompt-only control (FST-analogue): render config FROZEN at
             defaults; evolves only the free-text slots.

Scoring: scorer.py (validator-exact on D_harness, 2-rollout noise control,
noopFP +2pp / p95 1.3x guardrails vs the default baseline). Every proposer
call ledgered with token counts; every candidate's completion spend tracked
via the client ledger deltas.

State: results/<arm>/state.json (checkpoint per iteration — resumable);
results/cache-<modeltag>.jsonl (completion pair-cache, shared across arms).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "eval"))
sys.path.insert(0, str(HERE.parent / "synthetic-data"))

from harness_config import (DEFAULT_CONFIG, DEFAULT_TEXTS, SPACE,  # noqa: E402
                            validate_config, validate_texts, fingerprint,
                            stops_for)
from render import render_scenario, build_scoped_prompt              # noqa: E402
from scorer import (score_candidate, score_noop, guardrails,         # noqa: E402
                    selection_key, predict)
from proposer import Proposer                                         # noqa: E402
import server as S                                                    # noqa: E402

FAMILIES = ("rename_propagation", "pipe_rewrite", "format_propagation",
            "doc_sync", "na_rm_propagation")
NOOP_SUBSET_N = 120          # search-time guardrail subset (docstring floor)
NOOP_SUBSET_SEED = 4242
MIGRATION_ITERS = (7, 10, 13)

GEPA_BANK = [
    "Complete the edited R region; apply the same change pattern as the recent edit.",
    "Predict the full updated region only. No explanations, no markers.",
    "Follow the file's style. One precise edit.",
    "Repeat the demonstrated transformation on the current region.",
]


def load_d_harness() -> list[dict]:
    rows = []
    for line in open(HERE / "data" / "d_harness.jsonl"):
        r = json.loads(line)
        r["id"] = hashlib.sha1(r["_prompt"].encode()).hexdigest()[:12]
        rows.append(r)
    return rows


_NOOP_CASES = None


def load_noop_cases(subset_n: int = NOOP_SUBSET_N) -> list:
    """The eval_noop_fp construction (same classes), seeded subset for the
    search-time guardrail; the verdict battery uses the FULL set."""
    global _NOOP_CASES
    if _NOOP_CASES is None:
        import eval_noop_fp
        _NOOP_CASES = eval_noop_fp.build_cases(corpus_n=30, seed=7)
    cases = list(_NOOP_CASES)
    if subset_n and len(cases) > subset_n:
        rng = random.Random(NOOP_SUBSET_SEED)
        idx = sorted(rng.sample(range(len(cases)), subset_n))
        cases = [cases[i] for i in idx]
    return cases


def failure_digest(rows, rollouts_by_row, cfg, limit=10) -> list[str]:
    """Compact per-example outcome lines for the proposer prompt."""
    from eval_scenarios import validator_verdict
    out, counts = [], {"shape": 0, "transform": 0, "unstable": 0, "error": 0}
    for row in rows:
        rolls = rollouts_by_row[row["id"]]
        preds = [predict(r["text"], row, cfg) for r in rolls]
        verd = []
        for p in preds:
            try:
                verd.append(bool(validator_verdict(row, p)[0]))
            except Exception:
                verd.append(False)
        if verd[0] and verd[1]:
            continue
        if verd[0] != verd[1]:
            counts["unstable"] += 1
            continue
        ok, kind, reason = validator_verdict(row, preds[0])
        counts[kind or "transform"] += 1
        if len(out) < limit:
            tgt = (row["region_new"][0] or "")[:70]
            prd = (preds[0][0] if preds else "")[:70]
            npl = len(preds[0]) if preds else 0
            out.append(f"- [{row['family']}] {kind}: {npl} line(s); "
                       f"pred={prd!r} target={tgt!r}")
    summary = (f"fail mix: shape={counts['shape']} transform={counts['transform']} "
               f"unstable={counts['unstable']}")
    return [summary] + out


class Evaluator:
    """Evaluates candidate batches against D_harness + the noopFP subset.

    MULTI-SERVER SHARDING: one PairClient per server port. Rows (and noop
    cases) are split by index parity across servers; each shard processes
    its rows with the iteration's candidates INNER (row-major per shard),
    so (a) each server keeps the KV prefix reuse across candidates of the
    same row and (b) both servers are loaded concurrently (the latency
    guardrail's baseline is measured under the same co-running regime).
    The 2-rollout pair stays sequential on one server per key.
    """

    def __init__(self, rows, noop_cases, clients, baseline=None):
        if isinstance(clients, S.PairClient):
            clients = [clients]
        self.rows, self.noop_cases = rows, noop_cases
        self.clients, self.baseline = clients, baseline

    def _delta(self):
        return dict(self.client_ledger())

    def client_ledger(self):
        led = dict(new_completions=0, pair_reuses=0, cache_entries=0)
        for c in self.clients:
            for k, v in c.ledger.items():
                led[k] += v
        return led

    def evaluate_batch(self, cands: list) -> list[dict]:
        """Evaluate one iteration's candidates together (sharded across
        servers). Per-candidate completion attribution: a key's completions
        are charged to the FIRST candidate in the batch that needed it;
        later ones count as pair reuses (the archive semantics)."""
        pre_keys = set()
        for c in self.clients:
            pre_keys.update(c.cache.keys())
        n_shard = len(self.clients)
        sc_rolls = [dict() for _ in cands]
        np_rolls = [dict() for _ in cands]
        sc_keys = [set() for _ in cands]
        np_keys = [set() for _ in cands]

        def run_shard(ci):
            client = self.clients[ci]
            for j, row in enumerate(self.rows):
                if j % n_shard != ci:
                    continue
                for i, (cfg, texts, _tag) in enumerate(cands):
                    prompt, _meta = render_scenario(row, cfg, texts)
                    sc_rolls[i][row["id"]] = client.pair(
                        prompt, stops_for(cfg), cfg["max_tokens"])
                    sc_keys[i].add(client.key(prompt, stops_for(cfg),
                                              cfg["max_tokens"]))
            for j, c in enumerate(self.noop_cases):
                if j % n_shard != ci:
                    continue
                for i, (cfg, texts, _tag) in enumerate(cands):
                    prompt, _meta = build_scoped_prompt(
                        c.lines, c.cursor_line, c.cursor_char, c.rel_path,
                        cfg, texts)
                    np_rolls[i][c.id] = client.pair(
                        prompt, stops_for(cfg), cfg["max_tokens"])
                    np_keys[i].add(client.key(prompt, stops_for(cfg),
                                              cfg["max_tokens"]))

        if n_shard == 1:
            run_shard(0)
        else:
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(n_shard) as ex:
                list(ex.map(run_shard, range(n_shard)))

        entries, claimed = [], set()
        for i, (cfg, texts, tag) in enumerate(cands):
            score = score_candidate(self.rows, sc_rolls[i], cfg)
            noop = score_noop(self.noop_cases, np_rolls[i], cfg)
            guard = guardrails(score, noop, self.baseline) if self.baseline else None
            completions = reuses = 0
            for keys in (sc_keys[i], np_keys[i]):
                for k in keys:
                    if k in pre_keys or k in claimed:
                        reuses += 1
                    else:
                        claimed.add(k)
                        completions += 2
            entries.append(dict(
                tag=tag, cfg=cfg, texts=texts, fp=fingerprint(cfg, texts),
                exact_pass=score["exact_pass"], unstable_frac=score["unstable_frac"],
                families=score["families"], p95_latency_s=score["p95_latency_s"],
                mean_latency_s=score["mean_latency_s"], noop=noop, guard=guard,
                completions=completions, pair_reuses=reuses,
                digest=failure_digest(self.rows, sc_rolls[i], cfg),
            ))
        return entries

    def evaluate(self, cfg, texts=None, tag="") -> dict:
        texts = texts or DEFAULT_TEXTS
        return self.evaluate_batch([(validate_config(cfg), texts, tag)])[0]


# --- seeded fallback mutations (logged as origin="fallback") ---------------

def fallback_config(rng: random.Random, parents: list[dict]) -> dict:
    parent = dict(rng.choice(parents) if parents else DEFAULT_CONFIG)
    knobs = rng.sample(sorted(SPACE), k=rng.choice([1, 1, 2]))
    for k in knobs:
        others = [v for v in SPACE[k] if v != parent.get(k)]
        if others:
            parent[k] = rng.choice(others)
    try:
        return validate_config(parent)
    except ValueError:
        return dict(DEFAULT_CONFIG)


def fallback_gepa(rng: random.Random, parents: list[dict]) -> dict:
    parent = dict(rng.choice(parents) if parents else DEFAULT_TEXTS)
    if not parent.get("instruction_line"):
        parent["instruction_line"] = rng.choice(GEPA_BANK)
    elif rng.random() < 0.5:
        line = parent["instruction_line"]
        parent["instruction_line"] = (line.split(";")[0] if ";" in line
                                      else rng.choice(GEPA_BANK))
    if rng.random() < 0.35:
        parent["checklist_line"] = rng.choice(GEPA_BANK)
    elif parent.get("checklist_line"):
        parent["checklist_line"] = ""
    return validate_texts(parent)


# --- the driver --------------------------------------------------------------

def proposer_ctx(state: dict, kind: str, migration: bool) -> dict:
    ev = state["evaluated"]
    best = state["best_entry"]
    ranked = sorted(ev, key=lambda e: -e["exact_pass"])
    arch = ([dict(fp=e["fp"], exact=e["exact_pass"],
                  cfg=(e["cfg"] if kind == "config" else e["texts"]))
             for e in ranked[:8]])
    if state.get("seed"):
        arch.insert(0, dict(fp=state["seed"]["fp"],
                            exact=state["seed"]["exact_pass"],
                            cfg=(state["seed"]["cfg"] if kind == "config"
                                 else state["seed"]["texts"])))
    ctx = dict(best=(best["cfg"] if kind == "config" else best["texts"]),
               best_kind=("config" if kind == "config" else "texts"),
               best_exact=best["exact_pass"], best_unstable=best["unstable_frac"],
               best_p95=best["p95_latency_s"], best_noop=best["noop"]["proposal_rate"],
               archive=arch, outcomes=best.get("digest"))
    if kind == "config" and state.get("method") == "population" and migration:
        d0 = [e for e in ev if e.get("deme") == 0]
        d1 = [e for e in ev if e.get("deme") == 1]
        ctx["deme_hint"] = (
            f"MIGRATION iteration: demes are {{0: {len(d0)}, 1: {len(d1)}}} "
            "evaluated configs. At least one candidate MUST be a cross-deme "
            "remix (one parent from each deme); set \"parents\":[fpA,fpB] on it.")
    return ctx


def next_candidates(state, proposer, kind, it) -> list[dict]:
    """3 candidates: proposer's valid ones + seeded fallbacks, deduped."""
    rng = random.Random(f"{state['arm']}-{it}")
    seen = {e["fp"] for e in state["evaluated"]}
    if state.get("seed"):
        seen.add(state["seed"]["fp"])  # never re-evaluate the default config
    out, origins = [], []
    if proposer is not None:
        try:
            cands, raw = proposer.propose(
                kind, proposer_ctx(state, kind, it in MIGRATION_ITERS),
                state["arm"], it)
        except Exception as e:
            cands, raw = [], f"proposer-error: {e}"
        state.setdefault("proposer_raw", {})[str(it)] = (raw or "")[:400]
        for c in cands:
            fp = fingerprint(c) if kind == "config" else fingerprint(DEFAULT_CONFIG, c)
            if fp not in seen:
                out.append(c)
                origins.append("proposer")
                seen.add(fp)
    while len(out) < 3:
        if kind == "config":
            parents = [dict(e["cfg"]) for e in state["evaluated"]] or [DEFAULT_CONFIG]
            if state.get("method") == "population":
                # migration fallbacks pick parents across demes
                d0 = [e for e in state["evaluated"] if e.get("deme") == 0]
                d1 = [e for e in state["evaluated"] if e.get("deme") == 1]
                if it in MIGRATION_ITERS and d0 and d1:
                    parents = [dict(rng.choice(d0)["cfg"]), dict(rng.choice(d1)["cfg"])]
            c = fallback_config(rng, parents)
            fp = fingerprint(c)
            if fp in seen:
                continue
        else:
            parents = [dict(e["texts"]) for e in state["evaluated"]] or [DEFAULT_TEXTS]
            c = fallback_gepa(rng, parents)
            fp = fingerprint(DEFAULT_CONFIG, c)
            if fp in seen:
                continue
        out.append(c)
        origins.append("fallback")
        seen.add(fp)
    return out, origins


def run_arm(arm: str, method: str, kind: str, iterations: int, port: int,
            model: str, results_dir: Path, proposer: Proposer | None,
            evaluator: Evaluator, resume: bool = True, seed: dict | None = None):
    state_path = results_dir / arm / "state.json"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    if resume and state_path.exists():
        state = json.loads(state_path.read_text())
        print(f"[{arm}] resume: {state['iterations_done']} iters, "
              f"{len(state['evaluated'])} evaluated", flush=True)
    else:
        state = dict(arm=arm, method=method, kind=kind, iterations_done=0,
                     evaluated=[], started=time.strftime("%FT%T"), model=model,
                     baseline=evaluator.baseline, seed=seed,
                     best_entry=seed)
    for it in range(state["iterations_done"] + 1, iterations + 1):
        t0 = time.time()
        cands, origins = next_candidates(state, proposer, kind, it)
        batch = []
        for cand in cands:
            if kind == "config":
                batch.append((cand, DEFAULT_TEXTS, f"{arm}-it{it}"))
            else:
                batch.append((DEFAULT_CONFIG, cand, f"{arm}-it{it}"))
        entries = evaluator.evaluate_batch(batch)
        for cand, origin, entry in zip(cands, origins, entries):
            entry.update(origin=origin, iter=it)
            if method == "population" and kind == "config":
                entry["deme"] = _child_deme(state, cand, origin, it)
            state["evaluated"].append(entry)
            _update_best(state, entry)
            print(json.dumps(dict(arm=arm, iter=it, fp=entry["fp"],
                                  origin=origin, exact=entry["exact_pass"],
                                  unstable=entry["unstable_frac"],
                                  p95=entry["p95_latency_s"],
                                  noop=entry["noop"]["proposal_rate"],
                                  guard=entry["guard"],
                                  completions=entry["completions"],
                                  reuses=entry["pair_reuses"])), flush=True)
        state["iterations_done"] = it
        state["proposer_totals"] = (proposer.totals if proposer else
                                    state.get("proposer_totals"))
        state_path.write_text(json.dumps(state, indent=1))
        print(f"[{arm}] iter {it}/{iterations} done in "
              f"{round(time.time() - t0)}s; best exact="
              f"{state['best_entry']['exact_pass']}", flush=True)
    state_path.write_text(json.dumps(state, indent=1))
    return state


def _child_deme(state, cand, origin, it) -> int:
    """Deme inheritance: deme 0 by default (the seed's); from iteration 8 on,
    children of parents evaluated under deme 1 inherit 1 (the proposer is
    free to remix across demes; migration iters force it)."""
    ev = state["evaluated"]
    if not ev:
        return 0
    if it <= 6:
        return ev[-1].get("deme", 0)
    best1 = [e for e in ev if e.get("deme") == 1]
    if best1 and max(e["exact_pass"] for e in best1) >= max(
            e["exact_pass"] for e in ev if e.get("deme", 0) == 0):
        return 1
    return 0


def _update_best(state, entry):
    """Best = best selection key among guardrail-PASSING entries (revert on
    regression is implicit: the best only changes when beaten)."""
    if entry["guard"] and not (entry["guard"]["noop_ok"] and entry["guard"]["latency_ok"]):
        if state["best_entry"] is None:
            state["best_entry"] = entry  # degenerate: baseline itself failing
        return
    cur = state["best_entry"]
    if cur is None or selection_key(entry) > selection_key(cur):
        state["best_entry"] = entry


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm", required=True,
                    choices=["baseline", "hill", "population", "gepa", "smoke"])
    ap.add_argument("--iters", type=int, default=13)
    ap.add_argument("--port", type=int, default=S.H1_PORT)
    ap.add_argument("--model", default=str(
        HERE.parent / "models" / "sft_v7_minicpm5-Q8_0.gguf"))
    ap.add_argument("--results", default=str(HERE / "results"))
    ap.add_argument("--no-server", action="store_true",
                    help="attach to an already-answering server (smoke)")
    ap.add_argument("--no-proposer", action="store_true",
                    help="fallback-mutation only (smoke)")
    ap.add_argument("--rows", type=int, default=0,
                    help="cap D_harness rows (smoke; 0 = all 256)")
    ap.add_argument("--noop-cap", type=int, default=0,
                    help="cap noopFP guardrail rows (smoke; 0 = full subset)")
    ap.add_argument("--ports", default=str(S.H1_PORT),
                    help="comma-separated server ports (sharded by row parity)")
    args = ap.parse_args()

    results = Path(args.results)
    results.mkdir(parents=True, exist_ok=True)
    model_tag = Path(args.model).stem.replace("-Q8_0", "")
    cache_path = results / f"cache-{model_tag}.jsonl"

    rows = load_d_harness()
    if args.rows:
        rows = rows[:args.rows]
    noop_cases = load_noop_cases()
    if args.noop_cap:
        noop_cases = noop_cases[:args.noop_cap]
    ports = [int(p) for p in str(args.ports).split(",")]
    fresh = args.arm == "baseline"  # baseline re-measures latency fresh
    clients = [S.PairClient(p, fresh=fresh) for p in ports]
    for c in clients:
        c.load(cache_path)

    srv = []
    if not args.no_server:
        for p in ports:
            s = S.Server(args.model, port=p,
                         log_path=results / f"llama-server-h1-{p}.log")
            s.start()
            srv.append(s)
            print(f"[server] up on {p} (pid {s.proc.pid})", flush=True)
    try:
        evaluator = Evaluator(rows, noop_cases, clients)
        base_path = results / "baseline.json"
        if args.arm == "baseline":
            t0 = time.time()
            base = evaluator.evaluate(DEFAULT_CONFIG, DEFAULT_TEXTS, "baseline")
            base["wall_s"] = round(time.time() - t0)
            base["ports"] = ports
            base_path.write_text(json.dumps(base, indent=1))
            print(json.dumps({k: base[k] for k in
                              ("exact_pass", "unstable_frac", "p95_latency_s",
                               "completions", "wall_s")}, indent=1))
        else:
            if not base_path.exists():
                sys.exit("run --arm baseline first")
            base = json.loads(base_path.read_text())
            baseline = dict(p95_latency_s=base["p95_latency_s"],
                            noop_rate=base["noop"]["proposal_rate"])
            evaluator.baseline = baseline
            seed = dict(tag="seed", cfg=DEFAULT_CONFIG, texts=DEFAULT_TEXTS,
                        fp=fingerprint(DEFAULT_CONFIG),
                        exact_pass=base["exact_pass"],
                        unstable_frac=base["unstable_frac"],
                        p95_latency_s=base["p95_latency_s"],
                        noop=base["noop"], digest=base.get("digest"),
                        origin="seed", iter=0)
            proposer = None
            if args.arm != "smoke":
                proposer = Proposer(results / f"{args.arm}" / "proposer_ledger.jsonl")
            method = {"hill": "hill", "population": "population",
                      "gepa": "gepa", "smoke": "smoke"}[args.arm]
            kind = "gepa" if args.arm == "gepa" else "config"
            state = run_arm(args.arm, method, kind, args.iters, ports,
                            args.model, results, proposer, evaluator,
                            seed=seed)
            print(json.dumps(dict(
                arm=args.arm, best=state["best_entry"]["fp"],
                best_exact=state["best_entry"]["exact_pass"],
                proposer_totals=state.get("proposer_totals")), indent=1))
    finally:
        S.dump_all(clients, cache_path)
        for s in srv or []:
            s.stop()
        print("[server] stopped", flush=True)


if __name__ == "__main__":
    main()
