"""Task 6 (E2) — matched completion-masked SFT post-stage (post-persistence
test): does the best-vs-worst PRETRAIN-arm gap survive an identical light
SFT? (Puro: gains persisted, +1.6-2.0pp.)

Subcommands (CPU build; train/eval are GPU, claim per protocol):
  build   scenario jsonl -> tokenized 1025-wide blocks + loss mask
          (loss ONLY on region_new tokens; prompt/pad masked out).
          Rows: prefix (unmasked context) + region_new (masked target);
          suffix appended unmasked when it fits (free context). Rows
          longer than 1025 tok are LEFT-truncated so the completion is
          always fully inside. Deterministic holdout: every 8th row by
          md5. Output: e2_sft/ blocks.npy (N,1025) int32, mask.npy
          (N,1024) bool, meta.json. A2 32K tokenizer
          (datasets/a2_tokenizer_v1) — same vocab as the POC arms.
  train   load an arm ckpt (poc_twin final.pt), AdamW post-stage,
          IDENTICAL data/order/seed for every arm (sequential blocks,
          no shuffle — matched by construction). Saves final ckpt.
  eval    held-out completion CE (nats per MASKED token) for a ckpt —
          the E2 readout (before/after, best vs worst arm).

GPU discipline: POC_MEM_FRACTION + LaunchGate; blocks are consumed
sequentially with memmap (RAM-disciplined).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
POC_TWIN = os.path.abspath(os.path.join(HERE, "..", "poc_twin"))
sys.path.insert(0, POC_TWIN)

SEQ = 1025  # block width; targets = positions 1..1024

SCENARIO_FILES = [
    "comment_drafting.jsonl",
    "comment_insert.jsonl",
    "comment_to_code_synthetic.jsonl",
    "doc_sync.jsonl",
    "mid_roxygen.jsonl",
    "format_propagation.jsonl",
]
TOK_PATH = "/mnt/h/sepalith/datasets/a2_tokenizer_v1/tokenizer.json"


def _field_text(v):
    """Scenario fields are stringified lists (or plain strings)."""
    if isinstance(v, str) and v.startswith("["):
        try:
            import ast
            return "\n".join(ast.literal_eval(v))
        except (ValueError, SyntaxError):
            return v
    if isinstance(v, list):
        return "\n".join(v)
    return str(v)


def _row_parts(row):
    return (_field_text(row.get("prefix", "")),
            _field_text(row.get("region_new", row.get("region_old", ""))),
            _field_text(row.get("suffix", "")))


def build(argv=None):
    ap = argparse.ArgumentParser(description=build.__doc__)
    ap.add_argument("--src", default="/mnt/h/sepalith/datasets/scenarios_v1")
    ap.add_argument("--files", nargs="*", default=SCENARIO_FILES)
    ap.add_argument("--out", default="/tmp/poc_cma/e2_sft")
    ap.add_argument("--max-tokens", type=float, default=30e6)
    ap.add_argument("--holdout-mod", type=int, default=8)
    args = ap.parse_args(argv)
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(TOK_PATH)
    assert tok.get_vocab_size() <= 32768 + 64, tok.get_vocab_size()

    os.makedirs(args.out, exist_ok=True)

    # ---- pass 1: tokenize every row -> (ids, flags) stream pieces ----
    pieces_train, pieces_hold = [], []
    per_file = {}
    total = 0
    for fn in args.files:
        path = os.path.join(args.src, fn)
        n = 0
        with open(path, errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue  # torn line (live-appended files)
                pre, comp, suf = _row_parts(row)
                if not comp:
                    continue
                ids = tok.encode(pre + comp + suf,
                                 add_special_tokens=False).ids
                npre = len(tok.encode(pre, add_special_tokens=False).ids)
                ncomp = len(tok.encode(comp, add_special_tokens=False).ids)
                if npre + ncomp >= SEQ:
                    over = npre + ncomp + 1 - SEQ
                    if over > npre:  # LEFT-truncate the prompt only
                        over = npre
                    ids, npre2 = ids[over:], npre - over
                else:
                    npre2 = npre
                # boundary merges can make the combined encoding SHORTER
                # than npre+ncomp — clamp the mask to the actual ids
                m = [False] * min(npre2, len(ids))
                rest = len(ids) - npre2
                if rest > 0:
                    m += [True] * min(ncomp, rest)
                    m += [False] * (rest - min(ncomp, rest))
                assert len(m) == len(ids)
                row_id = hashlib.md5((fn + line[:200]).encode()).hexdigest()
                if int(row_id[:8], 16) % args.holdout_mod == 0:
                    pieces_hold.append((ids, m))
                else:
                    pieces_train.append((ids, m))
                    total += len(ids)
                    if total >= args.max_tokens:
                        break
                n += 1
        per_file[fn] = n
        print(f"[e2-build] {fn}: {n} rows", flush=True)
        if total >= args.max_tokens:
            break

    # ---- pass 2: pack pieces into exactly-SEQ-wide blocks ----
    def pack(pieces):
        blocks, masks = [], []
        cur_ids, cur_m = [], []
        for ids, m in pieces:
            i = 0
            while i < len(ids):
                take = SEQ - len(cur_ids)
                cur_ids += ids[i:i + take]
                cur_m += m[i:i + take]
                i += take
                if len(cur_ids) == SEQ:
                    blocks.append(cur_ids)
                    masks.append(cur_m[1:])  # targets = slots 1..SEQ-1
                    cur_ids, cur_m = [], []
        if len(cur_ids) > 1:  # pad out the final partial block
            cur_ids += [0] * (SEQ - len(cur_ids))
            cur_m += [False] * (SEQ - len(cur_m))
            blocks.append(cur_ids)
            masks.append(cur_m[1:])
        return blocks, masks

    train_b, train_m = pack(pieces_train)
    hold_b, hold_m = pack(pieces_hold)
    np.save(os.path.join(args.out, "train_blocks.npy"),
            np.array(train_b, dtype=np.int32).reshape(-1, SEQ))
    np.save(os.path.join(args.out, "train_mask.npy"),
            np.array(train_m, dtype=bool).reshape(-1, SEQ - 1))
    np.save(os.path.join(args.out, "holdout_blocks.npy"),
            np.array(hold_b, dtype=np.int32).reshape(-1, SEQ))
    np.save(os.path.join(args.out, "holdout_mask.npy"),
            np.array(hold_m, dtype=bool).reshape(-1, SEQ - 1))
    meta = dict(train_rows=len(train_b), holdout_rows=len(hold_b),
                train_tokens=len(train_b) * (SEQ - 1),
                masked_train_tokens=int(np.array(train_m, dtype=bool).sum()),
                files=per_file, tokenizer=TOK_PATH, seq=SEQ)
    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print(json.dumps(meta, indent=1), flush=True)
    return 0


def _run_batch_ce(model, x, mk):
    """Per-sequence sum nats over masked positions + masked counts."""
    import torch
    import torch.nn.functional as F
    h = model.trunk(x[:, :-1], probe=False)
    logits = F.linear(h, model.embed.weight)
    lg = logits.view(-1, logits.size(-1)).float()
    tg = x[:, 1:].reshape(-1)
    per = F.cross_entropy(lg, tg, reduction="none").view(x.size(0), -1)
    per = per * mk
    return per.sum(dim=1).detach(), mk.sum(dim=1)


def train(argv=None):
    ap = argparse.ArgumentParser(description=train.__doc__)
    ap.add_argument("--data", default="/tmp/poc_cma/e2_sft")
    ap.add_argument("--ckpt", required=True, help="arm final.pt")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--micro-bs", type=int, default=8)
    ap.add_argument("--log-every", type=int, default=10)
    args = ap.parse_args(argv)
    import torch
    import train as T  # poc_twin trainer (LaunchGate etc.)
    from model import TinyGQA

    T.LaunchGate().wait()
    torch.cuda.set_per_process_memory_fraction(
        float(os.environ.get("POC_MEM_FRACTION", "0.42")))
    torch.manual_seed(1273)

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model = TinyGQA(ck["cfg"]).cuda()
    model.load_state_dict({k: v.float() for k, v in ck["model"].items()})
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.wd)
    blocks = np.load(os.path.join(args.data, "train_blocks.npy"),
                     mmap_mode="r")
    mask = np.load(os.path.join(args.data, "train_mask.npy"), mmap_mode="r")
    n = len(blocks)
    accum = 64 // args.micro_bs  # 512-block step = 524288 tokens
    logf = open(f"/tmp/poc_cma/e2_{args.tag}.jsonl", "a")
    step, t0 = 0, __import__("time").time()
    i = 0
    opt.zero_grad()
    wsum = 0.0
    while i < n:
        got = 0
        while got < 512 and i < n:
            xb = torch.from_numpy(
                np.asarray(blocks[i:i + args.micro_bs]).astype(np.int64)).cuda()
            mb = torch.from_numpy(
                np.asarray(mask[i:i + args.micro_bs])).cuda()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                nats, cnt = _run_batch_ce(model, xb, mb)
                loss = nats.sum() / max(1.0, cnt.sum().float())
            (loss / accum).backward()
            wsum += float(nats.sum())
            got += args.micro_bs
            i += args.micro_bs
        for g in opt.param_groups:
            g["lr"] = (args.lr * min(1.0, (step + 1) / args.warmup))
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad()
        step += 1
        if step % args.log_every == 0:
            rec = dict(step=step, rows=i, loss=round(wsum, 1),
                       elapsed_s=round(__import__("time").time() - t0, 1),
                       tag=args.tag)
            logf.write(json.dumps(rec) + "\n")
            logf.flush()
            print(json.dumps(rec), flush=True)
    out = os.path.join("/tmp/poc_twin", f"ckpt_e2_{args.tag}")
    os.makedirs(out, exist_ok=True)
    torch.save(dict(step=ck["step"], cfg=ck["cfg"], args=vars(args),
                    base_ckpt=args.ckpt,
                    model={k: v.bfloat16() for k, v in
                           model.state_dict().items()}),
               os.path.join(out, "final.pt"))
    print(f"[e2-train] done -> {out}/final.pt", flush=True)
    return 0


def evaluate(argv=None):
    ap = argparse.ArgumentParser(description=evaluate.__doc__)
    ap.add_argument("--data", default="/tmp/poc_cma/e2_sft")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--json", default=None)
    args = ap.parse_args(argv)
    import torch
    import train as T
    from model import TinyGQA

    T.LaunchGate().wait()
    torch.cuda.set_per_process_memory_fraction(
        float(os.environ.get("POC_MEM_FRACTION", "0.42")))
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model = TinyGQA(ck["cfg"]).cuda().eval()
    model.load_state_dict({k: v.float() for k, v in ck["model"].items()})
    blocks = np.load(os.path.join(args.data, "holdout_blocks.npy"))
    mask = np.load(os.path.join(args.data, "holdout_mask.npy"))
    nats, toks = 0.0, 0
    with torch.no_grad():
        for i in range(0, len(blocks), 8):
            xb = torch.from_numpy(blocks[i:i + 8].astype(np.int64)).cuda()
            mb = torch.from_numpy(mask[i:i + 8]).cuda()
            a, b = _run_batch_ce(model, xb, mb)
            nats += float(a.sum())
            toks += int(b.sum())
    out = dict(tag=args.tag, ckpt=args.ckpt,
               holdout_completion_ce=nats / max(1, toks),
               masked_tokens=toks)
    js = args.json or f"/tmp/poc_cma/e2_eval_{args.tag}.json"
    with open(js, "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out), flush=True)
    return 0


def main(argv=None):
    sub = {"build": build, "train": train, "eval": evaluate}
    if len(sys.argv) < 2 or sys.argv[1] not in sub:
        sys.exit(f"usage: e2_sft.py {{{'|'.join(sub)}}} ...")
    return sub[sys.argv[1]](sys.argv[2:])


if __name__ == "__main__":
    sys.exit(main())
