"""C05: author DEV-only finish_block base functions.

The LLM-written half of DEV250's finish_block quota. It reuses TRAIN's
authoring path (experiments/synthetic-data/finish_block_author.py: prompt,
checklist gates, derive_rows -> rules_finish_block.derive_all) with two
changes:

- 20 new domains that match none of the 24 `author:*` groups in the DAT-02
  split (TRAIN, DEV or final), so every domain is a new DEV-only group;
- five backends, one per model, so each domain gets one function from each.

    uv run --no-project python author_dev_finish.py OUT_DIR [--per-domain 5]

Writes OUT_DIR/bases.jsonl (one accepted function per line), rows.jsonl
(all derived rows), rejects.jsonl and stats.json. Resumable: finished job
ids are skipped.
"""

import argparse
import json
import random
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
SYNTH = REPO / "experiments/synthetic-data"
sys.path.insert(0, str(SYNTH))

import finish_block_author as A  # noqa: E402
import rewrite_author_zai as ZA  # noqa: E402
from cases.rules import load_rules  # noqa: E402

AUTH = Path.home() / ".local/share/opencode/auth.json"
GO = "https://opencode.ai/zen/go/v1"
ZAI_URL = "https://api.z.ai/api/coding/paas/v4/chat/completions"

# Domains absent from the split's author:* groups. One group per domain.
DEV_SUBTOPICS = {
    "astronomy/photometry": "aperture photometry sky annulus; airmass "
    "extinction correction; light-curve period folding; magnitude zero-point fit",
    "seismology": "STA/LTA trigger picking; Richter local magnitude; "
    "hypocentral distance from station picks; b-value from a catalog",
    "hydrology/rivers": "rating-curve discharge from stage; baseflow "
    "separation filter; flood frequency by Gumbel; low-flow Q95 index",
    "forestry/timber": "tree volume from DBH and height; stand basal area "
    "per hectare; site index from dominant height; thinning schedule",
    "fisheries/stock-assessment": "von Bertalanffy growth fit start values; "
    "catch-per-unit-effort standardization; length-weight conversion; "
    "surplus production yield",
    "linguistics/corpus": "type-token ratio windows; collocation log-"
    "likelihood; syllable counting heuristic; keyness against a reference",
    "archaeology/radiocarbon": "radiocarbon age to calibrated range lookup; "
    "stratigraphic phase ordering; sherd count density per context",
    "veterinary/herd-health": "somatic cell score from counts; body "
    "condition trend per animal; vaccination due-date table; mastitis flags",
    "food-science/sensory": "triangle test significance; panel mean "
    "hedonic scores; shelf-life from accelerated storage; water activity bins",
    "materials/fatigue": "S-N curve Basquin fit; rainflow cycle counting "
    "summary; Miner damage sum; hardness conversion table",
    "acoustics/room": "Sabine reverberation time; octave band SPL "
    "summation; noise dose from dBA logs; absorption coefficient averaging",
    "aviation/flight-ops": "density altitude; fuel burn per leg; crosswind "
    "component per runway; takeoff weight margin",
    "manufacturing/spc": "X-bar and R chart limits; Cpk from samples; "
    "Western Electric rule flags; defect Pareto table",
    "retail/inventory": "reorder point with safety stock; ABC "
    "classification by revenue; sell-through rate; stockout day counting",
    "elections/voting": "D'Hondt seat allocation; swing between two "
    "elections; turnout by precinct; ranked-choice elimination rounds",
    "bibliometrics": "h-index per author; citation half-life; "
    "co-authorship edge list; journal impact window counts",
    "brewing/fermentation": "original gravity to ABV; IBU by Tinseth; "
    "mash water volume; fermentation attenuation curve",
    "textiles/color": "CIE Lab to Delta E 2000; yarn count conversion; "
    "fabric shrinkage percent; dye recipe scaling",
    "hydrogeology/groundwater": "Theis drawdown with the well function; "
    "hydraulic gradient from piezometers; recharge by water-table fluctuation",
    "apiculture": "colony weight trend per hive; varroa mite drop rate; "
    "honey yield per apiary; swarm-risk flag from inspection records",
}


def _key(provider: str) -> str:
    return json.loads(AUTH.read_text())[provider]["key"]


class Backend:
    def __init__(self, name, model, protocol, provider, max_tokens=16000):
        self.name, self.model, self.protocol = name, model, protocol
        self.provider, self.max_tokens = provider, max_tokens
        self.session = str(uuid.uuid4())

    def _request(self, url, body):
        req = urllib.request.Request(url, json.dumps(body).encode(), {
            "Authorization": f"Bearer {_key(self.provider)}",
            "Content-Type": "application/json",
            "User-Agent": "curl/8.5.0",
            "x-opencode-session": self.session,
            "x-opencode-client": "sepalith",
        })
        with urllib.request.urlopen(req, timeout=900) as r:
            return json.loads(r.read())

    def complete(self, prompt: str) -> str:
        msg = [{"role": "user", "content": prompt}]
        last = None
        for attempt in range(4):
            try:
                if self.protocol == "responses":
                    p = self._request(f"{GO}/responses", {
                        "model": self.model,
                        "input": [{"role": "user", "content": [
                            {"type": "input_text", "text": prompt}]}],
                        "max_output_tokens": self.max_tokens,
                        "reasoning": {"effort": "low"},
                    })
                    return "".join(
                        c.get("text", "") for o in p.get("output", [])
                        if o.get("type") == "message"
                        for c in o.get("content", []) if isinstance(c, dict))
                url = ZAI_URL if self.provider == "zai-coding-plan" else f"{GO}/chat/completions"
                body = {"model": self.model, "messages": msg,
                        "max_tokens": self.max_tokens, "temperature": 0.95}
                if self.provider == "zai-coding-plan":
                    body["thinking"] = {"type": "enabled"}
                    body["reasoning_effort"] = "low"
                p = self._request(url, body)
                return p["choices"][0]["message"]["content"] or ""
            except (urllib.error.URLError, TimeoutError, KeyError, ValueError) as e:
                code = getattr(e, "code", None)
                body = e.read()[:300].decode("utf-8", "replace") if hasattr(e, "read") else ""
                last = f"{e!r} {body}"
                if code is not None and code < 500 and code != 429:
                    raise RuntimeError(f"{self.name}: {last}") from e
                time.sleep(15 * (attempt + 1))
        raise RuntimeError(f"{self.name}: retries exhausted; last: {last}")


BACKENDS = [
    Backend("zai-glm-5.3", "glm-5.3", "chat", "zai-coding-plan"),
    Backend("go-space-bunny-free", "space-bunny-free", "chat", "opencode-go"),
    Backend("go-muse-spark-1.3-contributor", "muse-spark-1.3-contributor",
            "responses", "opencode-go"),
    Backend("go-longcat-2.5-preview-free", "longcat-2.5-preview-free", "chat",
            "opencode-go"),
    Backend("go-mimo-v2.6-pro", "mimo-v2.6-pro", "chat", "opencode-go"),
]


A.SUBTOPICS.update(DEV_SUBTOPICS)


def prompt_for(cell, feedback=None):
    text = A.build_prompt(cell, feedback)
    # The DEV domains have no CRAN seed packages; drop that clause.
    return text.replace(" (in the spirit of the CRAN packages: )", "")


def jobs(per_domain: int, seed: int):
    out = []
    for d_i, domain in enumerate(sorted(DEV_SUBTOPICS)):
        for k in range(per_domain):
            rng = random.Random(f"{seed}:{domain}:{k}")
            cell = dict(domain=domain, seeds=[],
                        style=rng.choice(A.STYLES), length=rng.choice(A.LENGTHS),
                        roxygen=rng.choice(A.ROXY),
                        constructs=rng.sample(sorted(A.CONSTRUCTS),
                                              rng.choice((0, 1, 1, 2))),
                        comments=rng.choice(A.COMMENTS))
            backend = BACKENDS[(d_i + k) % len(BACKENDS)]
            out.append((f"{A._slug(domain)}:{k}", cell, backend))
    return out


def run_job(job_id, cell, backend):
    feedback, first_prompt, raw = None, "", ""
    for attempt in (1, 2):
        prompt = prompt_for(cell, feedback)
        first_prompt = first_prompt or prompt
        try:
            raw = backend.complete(prompt)
        except Exception as e:  # noqa: BLE001
            return dict(kind="backend_error", job=job_id, backend=backend.name,
                        error=repr(e)[:300])
        obj = ZA.extract_json_object(ZA.strip_fences(raw))
        fn_text = obj.get("function") if isinstance(obj, dict) else None
        if not isinstance(fn_text, str) or "function" not in fn_text:
            feedback = "the response was not the JSON object with the function"
            continue
        if str(obj.get("domain_echo", "")).strip().lower() != cell["domain"]:
            feedback = f"domain_echo must be {cell['domain']!r}"
            continue
        fn_text = "\n".join(ZA._norm_trim(fn_text))
        gates, fails = A.gate_sample(cell, fn_text)
        if not fails:
            rows, dstats = A.derive_rows(fn_text, cell, first_prompt,
                                         backend.name, backend.model)
            return dict(kind="accepted", job=job_id, backend=backend.name,
                        model=backend.model, cell=cell, fn_text=fn_text,
                        subtopic_used=obj.get("subtopic_used"), gates=gates,
                        attempts=attempt, base_id=dstats.get("base_id"),
                        prompt=first_prompt, rows=rows,
                        derive_stats={k: v for k, v in dstats.items()
                                      if k != "base_id"})
        feedback = f"checklist mismatch: {fails}"
    return dict(kind="rejected", job=job_id, backend=backend.name,
                reason=feedback, last_raw=raw[:3000])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    ap.add_argument("--per-domain", type=int, default=5)
    ap.add_argument("--seed", type=int, default=20260930)
    ap.add_argument("--workers-per-backend", type=int, default=2)
    args = ap.parse_args()
    load_rules()
    args.out.mkdir(parents=True, exist_ok=True)
    bases_p, rows_p = args.out / "bases.jsonl", args.out / "rows.jsonl"
    rejects_p = args.out / "rejects.jsonl"
    done = set()
    for p in (bases_p, rejects_p):
        if p.exists():
            done |= {json.loads(line)["job"] for line in p.open()
                     if json.loads(line).get("kind") != "backend_error"}
    todo = [j for j in jobs(args.per_domain, args.seed) if j[0] not in done]
    lock = threading.Lock()
    stats = Counter()

    def record(res):
        with lock:
            stats[(res["backend"], res["kind"])] += 1
            if res["kind"] == "accepted":
                rows = res.pop("rows")
                res["n_rows"] = len(rows)
                with bases_p.open("a") as f:
                    f.write(json.dumps(res) + "\n")
                with rows_p.open("a") as f:
                    for r in rows:
                        r = dict(r, job=res["job"], base_id=res["base_id"])
                        f.write(json.dumps(r) + "\n")
            else:
                with rejects_p.open("a") as f:
                    f.write(json.dumps(res) + "\n")
            print(res["kind"], res["job"], res["backend"], flush=True)

    pools = {b.name: ThreadPoolExecutor(args.workers_per_backend) for b in BACKENDS}
    futs = [pools[b.name].submit(lambda j=j, c=c, b=b: record(run_job(j, c, b)))
            for j, c, b in todo]
    for f in futs:
        f.result()
    summary = {f"{b}|{k}": v for (b, k), v in sorted(stats.items())}
    (args.out / "stats.json").write_text(json.dumps(dict(
        finished=time.strftime("%Y-%m-%dT%H:%M:%S"), jobs=len(todo),
        per_domain=args.per_domain, seed=args.seed, counts=summary,
        backends=[dict(name=b.name, model=b.model, protocol=b.protocol)
                  for b in BACKENDS]), indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
