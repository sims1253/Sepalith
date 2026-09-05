"""Rebuild /tmp/poc_twin/ladder/tokenizer_minicpm5.json (the /tmp copy is
volatile). Source = the HF MiniCPM5 tokenizer — the SAME tokenizer the
corpus was packed with (data_prep.py), so tokens-by-id are identical to the
night-ladder's GGUF-embedded copy by construction."""
import json, os

from transformers import AutoTokenizer

OUT = "/tmp/poc_twin/ladder/tokenizer_minicpm5.json"

t = AutoTokenizer.from_pretrained("openbmb/MiniCPM5-1B")
tokens = t.convert_ids_to_tokens(list(range(len(t))))   # incl. added specials
assert len(tokens) == 130560, len(tokens)
merges = json.loads(t.backend_tokenizer.to_str())["model"]["merges"]
# tokenizers>=0.21 serializes merges as [a, b] PAIRS; gguf wants "a b" strings
merges = [" ".join(m) if isinstance(m, (list, tuple)) else m for m in merges]
scal = dict(
    model="gpt2", pre="minicpm5",
    **{"tokenizer.ggml.bos_token_id": int(t.bos_token_id),
       "tokenizer.ggml.eos_token_id": int(t.eos_token_id),
       "tokenizer.ggml.unknown_token_id": int(t.unk_token_id),
       "tokenizer.ggml.padding_token_id": int(t.pad_token_id
                                              if t.pad_token_id is not None
                                              else t.eos_token_id)})
assert scal["tokenizer.ggml.eos_token_id"] == 1
os.makedirs(os.path.dirname(OUT), exist_ok=True)
json.dump(dict(tokens=tokens, merges=merges, scalars=scal), open(OUT, "w"))
print(f"wrote {OUT}: {len(tokens)} tokens, {len(merges)} merges, "
      f"pre={scal['pre']}, eos={scal['tokenizer.ggml.eos_token_id']}")
