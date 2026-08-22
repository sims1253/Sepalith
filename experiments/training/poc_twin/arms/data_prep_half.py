"""
Arm D data prep: corpus-half split of the twin-POC train corpus.

Seeded doc-level split of /tmp/poc_twin/train.jsonl (276,206 docs,
245.4M tokens): RandomState(SPLIT_SEED) permutation of doc indices,
keep the first half (138,103 docs) in ORIGINAL relative order, tokenize
+ pack with the parent data_prep.py functions (identical tokenizer,
eos separators, overlap-1 1025-blocks). Output:
  /tmp/poc_twin/half_train.jsonl           (the kept docs)
  /tmp/poc_twin/train_blocks_half.npy      (packed blocks, ~122.7M tokens)
  /tmp/poc_twin/meta_half.json
  /tmp/poc_twin/gen_prompts_train.json     (first-120-token prompts of the
                     first 10 half-corpus docs — IN-CORPUS for BOTH the
                     full-corpus and half-corpus models, so the
                     regurgitation canary is paired)
Held-out prompts for the canary = gen_prompts.json rows 10..19 (eval
corpus, unseen by both models).
"""
import json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from data_prep import pack_file, to_blocks  # noqa: E402

TMP = "/tmp/poc_twin"
SPLIT_SEED = 20260821
SRC = os.path.join(TMP, "train.jsonl")


def main():
    # 1) seeded half split at the doc level
    n_docs = sum(1 for _ in open(SRC, encoding="utf-8", errors="replace"))
    rng = np.random.RandomState(SPLIT_SEED)
    perm = rng.permutation(n_docs)
    keep = np.sort(perm[: n_docs // 2])           # original relative order
    keep_set = set(keep.tolist())
    out_jsonl = os.path.join(TMP, "half_train.jsonl")
    if not os.path.exists(out_jsonl):
        print(f"splitting: {len(keep)}/{n_docs} docs (seed {SPLIT_SEED})", flush=True)
        with open(SRC, encoding="utf-8", errors="replace") as f, \
                open(out_jsonl, "w", encoding="utf-8") as g:
            for i, line in enumerate(f):
                if i in keep_set:
                    g.write(line)
    else:
        print(f"{out_jsonl} exists, skipping split", flush=True)

    # 2) tokenize + pack identically to the parent prep
    out_npy = os.path.join(TMP, "train_blocks_half.npy")
    tokens, docs = pack_file(out_jsonl)
    blocks = to_blocks(tokens)
    np.save(out_npy, blocks)
    meta = dict(split_seed=SPLIT_SEED, half_docs=int(docs),
                half_tokens=int(len(tokens)), half_blocks=int(len(blocks)),
                full_docs=int(n_docs), seq=1025, eos_id=1)
    with open(os.path.join(TMP, "meta_half.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print(json.dumps(meta), flush=True)

    # 3) 10 in-corpus generation prompts (first 120 tokens of the first 10
    #    kept docs), same recipe as the parent's eval gen_prompts
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("openbmb/MiniCPM5-1B")
    prompts = []
    with open(out_jsonl, encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f):
            if i >= 10:
                break
            ids = tok(json.loads(line)["text"], add_special_tokens=False)["input_ids"][:120]
            prompts.append(ids)
    with open(os.path.join(TMP, "gen_prompts_train.json"), "w") as f:
        json.dump(prompts, f)
    print("wrote gen_prompts_train.json", flush=True)


if __name__ == "__main__":
    main()
