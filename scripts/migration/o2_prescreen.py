#!/usr/bin/env python3
"""O2 pre-screen: per-prompt pass rates under the frozen b4 base (k=16,
temp 1.0 full-support sampling matched to GRPO rollouts), admission band
solves in [3,13]/16. Serving path decided by a 256-prompt calibration leg
(Q8 llama-server vs HF bf16; fallback f16 server). No training here.

Zones: prepare (CPU) -> measure (GPU, 8h bound) -> evaluate (CPU).
Reward path verbatim from rl_smoke/run_eval: parse_pred('zeta2', ...) +
norm; solve := parsed prediction == normalized target lines.
"""
import argparse, hashlib, json, math, os, signal, sys, time, urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'experiments/eval'))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'experiments/training'))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts/migration'))
import rl_smoke
from run_eval import norm, parse_pred
from s1_gpu import GpuServer, DeadlineExceeded, check_offload, write

DATA = Path('/mnt/h/sepalith/datasets/sft_v7/train.jsonl')
HOLDOUT = Path('/mnt/h/sepalith/datasets/sft_v3/eval.jsonl')
BASE_HF = Path('/home/m0hawk/Documents/Sepalith/experiments/models/qwen3.5-2b-base-text-hf')
B4_HF = Path('/mnt/h/sepalith/runs/pft1_b4_merged')
Q8_GGUF = Path('/home/m0hawk/Documents/Sepalith/experiments/models/b4_qwen35_2b-Q8_0.gguf')
F16_GGUF = Path('/home/m0hawk/Documents/Sepalith/experiments/models/packaging_b4-f16.gguf')
OUTROOT = Path('/mnt/h/sepalith/datasets/rl_o2_prescreen')
K = 16
BAND = (3, 13)
MAXTOK = 192
CAL_N_PER_FAM = 64
QUOTA = dict(rl_smoke.FAMILY_QUOTA)
# refill margins (plan §1.3): +50% format/no_op/pipe, +100% rename
EXTENDED_QUOTA = {'rename_propagation': 2800, 'format_propagation': 2100,
                  'no_op': 525, 'pipe_rewrite': 225}
from concurrent.futures import ThreadPoolExecutor


def digest(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()


def solve_of(completion_text, target):
    pred = parse_pred('zeta2', completion_text)
    gt = norm(target.splitlines())
    return pred == gt


def pool_rows():
    """Verbatim guard path, extended quotas for refill walk."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(BASE_HF))
    rows, dstat = rl_smoke.build_dataset(tok, quotas=EXTENDED_QUOTA, seed=3407, data_path=DATA)
    return rows, dstat


def seed_base_for(fam, rank, fam_idx):
    return 3407 + 1000 * fam_idx[fam] + 16 * rank


def prepare(run, assets):
    rows, dstat = pool_rows()
    fams = sorted(QUOTA)
    counts = {f: sum(1 for r in rows if r['family'] == f) for f in fams}
    for f in fams:
        if counts[f] < QUOTA[f]:
            raise ValueError('extended pool smaller than quota for %s: %d' % (f, counts[f]))
    # eval-battery overlap assert (prompt hashes)
    import hashlib as _h
    ev = assets / 'cases.jsonl'
    if ev.is_file():
        eval_prompts = {json.loads(l)['prompt'] for l in ev.read_text().splitlines() if l.strip()}
        overlap = sum(1 for r in rows if r['prompt'] in eval_prompts)
        if overlap:
            raise ValueError('pool overlaps frozen eval battery: %d' % overlap)
    cal = []
    for f in fams:
        fr = [r for r in rows if r['family'] == f]
        cal.extend(fr[:CAL_N_PER_FAM])
    write(run / 'prepared.json', dict(
        families=fams, quotas=QUOTA, k=K, band=BAND, max_tokens=MAXTOK,
        cal_prompts=len(cal), pool_rows=len(rows), pool_counts=counts,
        draw_stats=dstat, data=str(DATA), holdout=str(HOLDOUT),
        base_hf=str(BASE_HF), b4_hf_sha256=digest(B4_HF / 'model.safetensors'),
        q8_gguf_sha256=digest(Q8_GGUF), f16_gguf_sha256=digest(F16_GGUF),
        max_seconds=36000,
        extended_quotas=EXTENDED_QUOTA,
        calibration_gate=dict(n_per_family=CAL_N_PER_FAM, signed_bias_max=0.05,
                              note='signed per-family bias gate; flips reported descriptively '
                                   '(independent k=16 noise floor ~0.14 mean|dp|)'),
        family_floor_guard='format_propagation admitted >= 70% of quota else ABORT flag',
        scope='O2 pre-screen under frozen b4 via HF rollouts (exact GRPO parity); '
              'admission band [3,13]/16 solves; no training; arms separate',
        serving_decision='HF-only: llama-server b10453 Content-only PEG rejects invalid-UTF-8 '
                         'temp-1.0 rollouts (r2-r4 evidence); no calibration leg needed'))


def one_completion(server, prompt, seed):
    # raw /completion endpoint: the OpenAI /v1/completions layer in this build
    # VALIDATES output format and 500s when temp-1.0 rollouts emit special
    # tokens after the UPDATED marker (post-marker junk is irrelevant to
    # parse_pred). The native endpoint returns raw content without checks.
    body = json.dumps(dict(prompt=prompt, n_predict=MAXTOK, temperature=1.0,
                           top_p=1.0, top_k=0, min_p=0.0, seed=seed,
                           stream=False)).encode()
    req = urllib.request.Request('http://127.0.0.1:%d/completion' % server.port,
                                 data=body, headers={'Content-Type': 'application/json'})
    started = time.monotonic()
    with urllib.request.urlopen(req, timeout=600) as r:
        data = json.load(r)
    if 'content' not in data:
        raise ValueError('native /completion returned no content')
    return dict(text=data['content'], stop_type=data.get('stop_type'),
                wall_s=time.monotonic() - started, seed=seed)


def completions_via_server(server, prompt, k, seed_base):
    with ThreadPoolExecutor(max_workers=16) as pool:
        return list(pool.map(lambda j: one_completion(server, prompt, seed_base + j), range(k)))


def calibration_hf(prompts, run):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(BASE_HF))
    model = AutoModelForCausalLM.from_pretrained(str(B4_HF), dtype=torch.bfloat16).cuda().eval()
    torch.manual_seed(3407)
    results = {}
    try:
        with torch.no_grad():
            for idx, p in enumerate(prompts):
                enc = tok(p, return_tensors='pt', add_special_tokens=False).to('cuda')
                gen = model.generate(**enc, do_sample=True, temperature=1.0, top_p=1.0, top_k=0,
                                     max_new_tokens=MAXTOK, num_return_sequences=K,
                                     pad_token_id=tok.eos_token_id)
                texts = tok.batch_decode(gen[:, enc['input_ids'].shape[1]:], skip_special_tokens=True)
                results[p] = texts
                if idx % 32 == 0:
                    print(json.dumps(dict(hf_cal=idx, of=len(prompts))), flush=True)
    finally:
        del model
        torch.cuda.empty_cache()
    return results


def measure(run, assets):
    """HF-only screen: exact GRPO-rollout parity (trl sampling semantics),
    one generate call per prompt with num_return_sequences=K.

    Serving decision history (2026-09-09, attempts r2-r4): llama-server b10453
    post-processes ALL completion output through a Content-only PEG that throws
    on invalid-UTF-8 byte-fallback samples (a legitimate part of the temp-1.0
    rollout distribution; --no-jinja/--reasoning-format none/logit-bias bans do
    not bypass it). The server path is therefore unusable for faithful
    pass-rate estimation; the HF path is exact by construction and needs no
    calibration leg.
    """
    p = json.loads((run / 'prepared.json').read_text())
    rows, _ = pool_rows()
    fams = p['families']
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(BASE_HF))
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(str(B4_HF), dtype=torch.bfloat16).cuda().eval()
    torch.manual_seed(3407)

    def expire(*_):
        raise DeadlineExceeded('O2 HF screen exceeded bound')
    old = signal.signal(signal.SIGALRM, expire); signal.alarm(p['max_seconds'])
    comp_path = run / 'completions.jsonl'
    prior = {}
    if comp_path.exists():
        for line in comp_path.read_text().splitlines():
            rec = json.loads(line)
            prior[rec['prompt_hash']] = rec['solves']
    try:
        with comp_path.open('a') as out, torch.no_grad():
            for fam in fams:
                fam_rows = [r for r in rows if r['family'] == fam]
                admitted = 0
                pending = []
                for rank, r in enumerate(fam_rows):
                    if admitted >= QUOTA[fam]:
                        break
                    ph = hashlib.sha1(r['prompt'].encode()).hexdigest()
                    if ph in prior:
                        if BAND[0] <= prior[ph] <= BAND[1]:
                            admitted += 1
                        continue
                    pending.append((rank, r, ph))
                    if len(pending) < 4 and rank < len(fam_rows) - 1 and admitted + len(pending) <= QUOTA[fam]:
                        continue
                    # batched generate: up to 4 prompts x K sequences (left-padded)
                    tok.padding_side = 'left'
                    enc = tok([x[1]['prompt'] for x in pending], return_tensors='pt',
                              padding=True, add_special_tokens=False).to('cuda')
                    seeds = []
                    for (rk, rr, _), pos in zip(pending, range(len(pending))):
                        seeds.append(seed_base_for(fam, rk, {f: i for i, f in enumerate(fams)}))
                    torch.manual_seed(seeds[0])
                    gen = model.generate(**enc, do_sample=True, temperature=1.0, top_p=1.0,
                                         top_k=0, max_new_tokens=MAXTOK, num_return_sequences=K,
                                         pad_token_id=tok.pad_token_id)
                    new_len = gen.shape[1] - enc['input_ids'].shape[1]
                    decoded = tok.batch_decode(gen[:, -new_len:], skip_special_tokens=True)
                    for bi, (rk, rr, phh) in enumerate(pending):
                        texts = decoded[bi * K:(bi + 1) * K]
                        solves = sum(solve_of(txt, rr['target']) for txt in texts)
                        prior[phh] = solves
                        out.write(json.dumps(dict(family=fam, prompt_hash=phh, rank=rk,
                                                  solves=solves, k=K), allow_nan=False) + chr(10))
                        if BAND[0] <= solves <= BAND[1]:
                            admitted += 1
                    out.flush()
                    pending = []
                    done_rank = rk
                    if done_rank % 40 == 0:
                        print(json.dumps(dict(family=fam, rank=done_rank, admitted=admitted,
                                              quota=QUOTA[fam])), flush=True)
                print(json.dumps(dict(family=fam, quota=QUOTA[fam], pool=len(fam_rows),
                                      admitted_total=admitted,
                                      exhausted=admitted < QUOTA[fam])), flush=True)
    finally:
        signal.alarm(0)
        del model
        torch.cuda.empty_cache()
        signal.signal(signal.SIGALRM, old)


def evaluate(run):
    import collections
    comp = [json.loads(l) for l in (run / 'completions.jsonl').read_text().splitlines()]
    fams = sorted(QUOTA)
    by_fam = collections.defaultdict(list)
    for r in comp:
        by_fam[r['family']].append(r)
    summary = {}
    abort = False
    for f in fams:
        rs = by_fam[f]
        in_band = sum(1 for r in rs if BAND[0] <= r['solves'] <= BAND[1])
        never = sum(1 for r in rs if r['solves'] == 0)
        unan = sum(1 for r in rs if r['solves'] == K)
        admitted_quota = min(in_band, QUOTA[f])
        floor_ok = True
        if f == 'format_propagation' and admitted_quota < 0.7 * QUOTA[f]:
            floor_ok = False
            abort = True
        summary[f] = dict(screened=len(rs), in_band=in_band, never_solved=never,
                          unanimous=unan, quota=QUOTA[f], admitted_quota=admitted_quota,
                          floor_guard_ok=floor_ok)
    outdir = OUTROOT; outdir.mkdir(parents=True, exist_ok=True)
    with (outdir / 'prescreen_v1.jsonl').open('w') as f:
        for r in comp:
            reason = None
            if r['solves'] == 0:
                reason = 'never_solved'
            elif r['solves'] == K:
                reason = 'unanimous'
            f.write(json.dumps(dict(family=r['family'], prompt_hash=r['prompt_hash'], k=r['k'],
                                    solves=r['solves'], p_hat=round(r['solves'] / r['k'], 4),
                                    admitted=BAND[0] <= r['solves'] <= BAND[1],
                                    exclusion_reason=reason, band=[x / K for x in BAND])) + '\n')
    prep = json.loads((run / 'prepared.json').read_text())
    cal = {'serving_path': 'HF-bf16-exact'}
    (outdir / 'summary.json').write_text(json.dumps(dict(per_family=summary, total_screened=len(comp),
                                                         band=BAND, k=K, abort_flag=abort), indent=2) + '\n')
    (outdir / 'provenance.json').write_text(json.dumps(dict(
        data=prep['data'], holdout=prep['holdout'], data_sha256=digest(DATA),
        holdout_sha256=digest(HOLDOUT), b4_hf=str(B4_HF), b4_hf_sha256=prep['b4_hf_sha256'],
        q8_gguf_sha256=prep['q8_gguf_sha256'], f16_gguf_sha256=prep['f16_gguf_sha256'],
        served_path=cal.get('serving_path'), quotas=QUOTA, extended_quotas=EXTENDED_QUOTA,
        seed=3407, k=K, band=BAND, sampling=dict(temperature=1.0, top_p=1.0, top_k=0, min_p=0.0,
                                                 max_tokens=MAXTOK),
        calibration=cal, timestamp=time.strftime('%F %T')), indent=2) + '\n')
    write(run / 'evaluation.json', dict(per_family=summary, total_screened=len(comp), abort_flag=abort))
    write(run / 'verdict.json', dict(verdict='O2-PRESCREEN-COMPLETE' if not abort else 'O2-PRESCREEN-ABORT-FLOOR',
                                     adoption='NOT-ASSESSED',
                                     boundary='Admission dataset written; the two GRPO arms are separate recipes. '
                                              'Pass-rate estimates are k=16 binomial samples (SE<=0.125).'))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('action', choices=['prepare', 'measure', 'evaluate'])
    ap.add_argument('--run', type=Path, required=True)
    ap.add_argument('--assets', type=Path)
    a = ap.parse_args()
    evaluate(a.run) if a.action == 'evaluate' else globals()[a.action](a.run, a.assets)
