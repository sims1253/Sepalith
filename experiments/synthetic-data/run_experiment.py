#!/usr/bin/env python3
"""Thinking-level experiment: does higher reasoning cost buy better synthetic R?

Arms: reasoning_effort low / high / max on glm-5.3 (coding endpoint).
Tasks: analyst_script + finish_block (fixed grid seeds, same inputs per arm).
Gate:  jsonschema -> R parse -> jarl (experiments/synthetic/validate.py).
Metrics: validity per layer, jarl warnings, length, diversity (distinct 3-grams,
unique defined function names), tokens, latency. Emits results/summary.json.

Usage:
  run_experiment.py --dry-run            # no API; fake model to test mechanics
  run_experiment.py --arms low,high      # subset
  run_experiment.py --n 50               # per arm per task (default 50)
"""
import argparse, json, os, random, re, statistics, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import grid
from validate import validate, ANALYST_SCHEMA, FINISH_SCHEMA

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
ENDPOINT = "https://api.z.ai/api/coding/paas/v4/chat/completions"
MAX_TOKENS = {"low": 1500, "high": 4000, "max": 6000}


def api_call(prompt, effort, max_retries=3):
    key = os.environ.get("ZAI_API_KEY")
    if not key:
        raise RuntimeError("ZAI_API_KEY not set (source ~/.zshrc)")
    body = json.dumps({
        "model": "glm-5.3",
        "thinking": {"type": "enabled"},
        "reasoning_effort": effort,
        "messages": [{"role": "user", "content": prompt}],
        "response_format": {"type": "json_object"},
        "max_tokens": MAX_TOKENS[effort],
        "temperature": 0.8,
    }).encode()
    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(ENDPOINT, data=body,
                headers={"Authorization": f"Bearer {key}",
                         "Content-Type": "application/json"})
            t0 = time.time()
            with urllib.request.urlopen(req, timeout=300) as r:
                data = json.loads(r.read())
            msg = data["choices"][0]["message"]
            usage = data.get("usage", {})
            return (msg.get("content") or "", attempt,
                    {"reasoning": usage.get("completion_tokens_details", {}).get("reasoning_tokens", 0),
                     "total": usage.get("total_tokens", 0)},
                    time.time() - t0)
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            time.sleep(3 * (attempt + 1))


def fake_call(prompt, effort, attempt_seed=0, code_key="code"):
    rng = random.Random(hash(prompt) ^ attempt_seed)
    code = "\n".join(
        [f"# synthetic {effort} snippet {rng.randint(1000,9999)}",
         "library(dplyr)",
         "df <- read.csv('data.csv')",
         "out <- df |> filter(!is.na(AVAL)) |>",
         "  group_by(TRTP) |>",
         "  summarise(n = n(), mean = mean(AVAL, na.rm = TRUE))",
         "print(out)"])
    time.sleep(0.05)
    return (json.dumps({"intent": f"dry-run {effort} intent {rng.randint(0,999)}",
                        code_key: code, "packages_used": ["dplyr"]}), 0,
            {"reasoning": 10, "total": 100}, 0.05)


def one(task, prompt, schema, code_key, effort, dry):
    try:
        raw, retry, usage, lat = (fake_call(prompt, effort, code_key=code_key) if dry
                                  else api_call(prompt, effort))
        obj = json.loads(raw)
        ok, layer, info, jw = validate(obj, schema, code_key)
        return dict(task=task, effort=effort, ok=ok, layer=layer, info=info[:120],
                    retry=retry, usage=usage, latency_s=round(lat, 2),
                    jarl_warnings=jw,
                    code=obj.get(code_key, ""), intent=obj.get("intent", ""))
    except Exception as e:
        return dict(task=task, effort=effort, ok=False, layer="transport",
                    info=str(e)[:120], retry=2, usage={"reasoning": 0, "total": 0},
                    latency_s=0.0, code="", intent="")


def diversity(codes):
    ngrams, fnames = set(), set()
    for c in codes:
        toks = re.findall(r"[A-Za-z_.][A-Za-z0-9_.]*", c)
        ngrams |= {tuple(toks[i:i+3]) for i in range(len(toks) - 2)}
        fnames |= set(re.findall(r"(?:function\s*\(|<-\s*function)", c)) and \
                  set(re.findall(r"([A-Za-z.][A-Za-z0-9.]*)\s*<-\s*function", c))
    return dict(distinct_3grams=len(ngrams),
                unique_defined_fns=len(fnames),
                mean_lines=round(statistics.mean(c.count("\n") + 1 for c in codes), 1) if codes else 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="low,high,max")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--concurrency", type=int, default=5)
    args = ap.parse_args()
    RESULTS.mkdir(exist_ok=True)
    log = open(RESULTS / "experiment.log", "a")

    # fixed inputs shared across arms
    rng = random.Random(42)
    jobs = []
    for i in range(args.n):
        c = grid.cell(rng)
        jobs.append(("analyst_script", grid.ANALYST_PROMPT.format(**c),
                     ANALYST_SCHEMA, "code"))
    # finish_block: real roxygen signatures from the ingested corpus, if available
    roxy = sample_real_roxygen(args.n)
    for i in range(args.n):
        r = roxy[i % len(roxy)] if roxy else fallback_roxygen(i)
        jobs.append(("finish_block", grid.FINISH_PROMPT.format(
            roxygen=r["roxygen"], signature=r["signature"], line_target=12),
            FINISH_SCHEMA, "body"))

    arms = args.arms.split(",")
    calls = [(t, p, s, k, e) for e in arms for (t, p, s, k) in jobs]
    print(f"{len(calls)} calls ({len(jobs)} tasks x {len(arms)} arms)"
          f"{' [DRY-RUN]' if args.dry_run else ''}", flush=True)
    t0 = time.time()
    results = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        for i, r in enumerate(ex.map(lambda a: one(*a, dry=args.dry_run), calls)):
            results.append(r)
            log.write(json.dumps({k: v for k, v in r.items() if k != "code"}) + "\n")
            if (i + 1) % 25 == 0:
                print(f"  {i+1}/{len(calls)} ({time.time()-t0:.0f}s)", flush=True)
    log.close()
    (RESULTS / "records.jsonl").write_text(
        "\n".join(json.dumps(r) for r in results) + "\n")

    summary = {}
    for e in arms:
        for t in ("analyst_script", "finish_block"):
            rs = [r for r in results if r["effort"] == e and r["task"] == t]
            if not rs:
                continue
            good = [r for r in rs if r["ok"]]
            summary[f"{e}/{t}"] = dict(
                n=len(rs),
                valid_rate=round(len(good) / len(rs), 3),
                fail_layers={l: sum(1 for r in rs if r["layer"] == l)
                             for l in ("json", "parse", "jarl", "transport")},
                mean_jarl_warnings=round(statistics.mean(
                    [r.get("jarl_warnings", 0) for r in rs]), 2),
                mean_reasoning_tokens=round(statistics.mean(
                    r["usage"]["reasoning"] for r in rs)),
                mean_total_tokens=round(statistics.mean(
                    r["usage"]["total"] for r in rs)),
                mean_latency_s=round(statistics.mean(r["latency_s"] for r in rs), 1),
                diversity=diversity([r["code"] for r in good]),
            )
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


def sample_real_roxygen(n):
    """Real roxygen headers + signatures from the normalized corpus (NAS)."""
    root = Path("/mnt/h/sepalith/normalized")
    out, rng = [], random.Random(7)
    pkgs = sorted(root.iterdir())
    rng.shuffle(pkgs)
    for pkgd in pkgs[:60]:
        rdir = pkgd / next(pkgd.iterdir()) / pkgd.name / "R"
        if not rdir.is_dir():
            continue
        for f in sorted(rdir.glob("*.R")):
            try:
                lines = f.read_text(errors="ignore").splitlines()
            except Exception:
                continue
            for i, l in enumerate(lines):
                if re.match(r"^\s*(\w[\w.]*\s*(=|<-)\s*function\s*\()", l):
                    j, roxy = i - 1, []
                    while j >= 0 and (lines[j].lstrip().startswith("#'") or not lines[j].strip()):
                        if lines[j].lstrip().startswith("#'"):
                            roxy.insert(0, lines[j])
                        j -= 1
                    sig = l
                    k = i + 1
                    while k < len(lines) and not sig.rstrip().endswith("{"):
                        sig += "\n" + lines[k]; k += 1
                    sig = sig.strip().rstrip("{").strip()
                    if roxy and len(" ".join(roxy)) > 80 and out.__len__() < 400:
                        out.append(dict(roxygen="\n".join(roxy[:14]), signature=sig[:400]))
                        break
        if len(out) >= n:
            break
    rng.shuffle(out)
    return out


def fallback_roxygen(i):
    return dict(
        roxygen="#' Summarize change from baseline by visit\n#'\n#' @param data ADaM-style data frame with AVAL, BASE, AVISIT, TRTP\n#' @return grouped summary data frame",
        signature=f"summarize_chg{i} <- function(data, by_visit = TRUE)")


if __name__ == "__main__":
    main()
