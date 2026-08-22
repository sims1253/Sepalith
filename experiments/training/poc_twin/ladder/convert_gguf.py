"""
TinyGQA checkpoint -> GGUF (llama arch) for llama.cpp serving.

The model is a clean Llama-shape transformer (pre-RMSNorm, SwiGLU, GQA,
RoPE theta 500k, tied embeddings) with ONE quirk: attention inner dim
(16 heads x 64 = 1024) exceeds d_model (768), so attention.key_length /
value_length are set explicitly (b10453 optional KVs, llama-arch.cpp).

Tied head is materialized as an UNTIED duplicate output.weight (numerically
identical; avoids relying on tie_word_embeddings support). Weights stored
F32 (ckpts already hold bf16 values — F32 keeps the conversion lossless).

Tokenizer copied from the sft_v7_minicpm5 GGUF (the MiniCPM5 tokenizer the
corpus was tokenized with): 130,560 tokens / pre=minicpm5 / eos id 1.
add_bos_token is set FALSE — the corpus was tokenized with
add_special_tokens=False, so a served BOS would be a distribution shift.

Usage: convert_gguf.py --ckpt .../final.pt --out .../arm.gguf
"""
import argparse, json, os, sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
POC = os.path.dirname(HERE)
sys.path.insert(0, POC)
from model import TinyGQA, model_config  # noqa: E402

import gguf  # noqa: E402

TOKJSON = "/tmp/poc_twin/ladder/tokenizer_minicpm5.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = model_config(**{k: v for k, v in ck["cfg"].items()})
    model = TinyGQA(cfg)
    model.load_state_dict({k: v.float() for k, v in ck["model"].items()})
    sd = model.state_dict()

    tj = json.load(open(TOKJSON))
    tokens, merges, scal = tj["tokens"], tj["merges"], tj["scalars"]
    n_vocab = len(tokens)
    assert n_vocab == cfg["vocab"] == 130560, n_vocab

    w = gguf.GGUFWriter(args.out, "llama")
    w.add_string("general.name", "sepalith-poc-tinygqa-fim-ladder")
    w.add_uint32("general.version", 1)
    w.add_uint32("general.file_type", 0)  # F32
    w.add_architecture()  # binds "llama" from constructor
    w.add_uint32("llama.vocab_size", n_vocab)
    w.add_uint32("llama.context_length", 4096)
    w.add_uint32("llama.embedding_length", cfg["d_model"])
    w.add_uint32("llama.block_count", cfg["n_layers"])
    w.add_uint32("llama.feed_forward_length", cfg["ffn_hidden"])
    w.add_uint32("llama.attention.head_count", cfg["n_q"])
    w.add_uint32("llama.attention.head_count_kv", cfg["n_kv"])
    w.add_uint32("llama.attention.key_length", cfg["head_dim"])   # 64 != 768/16
    w.add_uint32("llama.attention.value_length", cfg["head_dim"])
    w.add_float32("llama.attention.layer_norm_rms_epsilon", 1e-6)
    w.add_float32("llama.rope.freq_base", cfg["rope_theta"])
    w.add_uint32("llama.rope.dimension_count", cfg["head_dim"])

    w.add_string("tokenizer.ggml.model", scal["model"])            # gpt2
    w.add_string("tokenizer.ggml.pre", scal["pre"])                # minicpm5
    w.add_array("tokenizer.ggml.tokens", tokens)
    w.add_array("tokenizer.ggml.merges", merges)
    ttype = [3 if i in (scal["tokenizer.ggml.bos_token_id"],
                        scal["tokenizer.ggml.eos_token_id"]) else 1
             for i in range(n_vocab)]
    w.add_array("tokenizer.ggml.token_type", ttype)
    w.add_uint32("tokenizer.ggml.bos_token_id",
                 scal["tokenizer.ggml.bos_token_id"])
    w.add_uint32("tokenizer.ggml.eos_token_id",
                 scal["tokenizer.ggml.eos_token_id"])
    w.add_uint32("tokenizer.ggml.unknown_token_id",
                 scal["tokenizer.ggml.unknown_token_id"])
    w.add_uint32("tokenizer.ggml.padding_token_id",
                 scal["tokenizer.ggml.padding_token_id"])
    w.add_bool("tokenizer.ggml.add_bos_token", False)   # training had no BOS
    w.add_bool("tokenizer.ggml.add_sep_token", False)

    def add(name, t):
        w.add_tensor(name, t.detach().numpy().astype(np.float32, copy=False))

    add("token_embd.weight", sd["embed.weight"])
    add("output_norm.weight", sd["ln_f.weight"])
    add("output.weight", sd["embed.weight"])        # untied duplicate
    for i, b in enumerate(model.blocks):
        p = f"blk.{i}."
        add(p + "attn_norm.weight", b.ln1.weight)
        add(p + "attn_q.weight", b.attn.Wq.weight)
        add(p + "attn_k.weight", b.attn.Wk.weight)
        add(p + "attn_v.weight", b.attn.Wv.weight)
        add(p + "attn_output.weight", b.attn.Wo.weight)
        add(p + "ffn_norm.weight", b.ln2.weight)
        add(p + "ffn_gate.weight", b.Wg.weight)
        add(p + "ffn_up.weight", b.Wu.weight)
        add(p + "ffn_down.weight", b.Wd.weight)

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"wrote {args.out} ({os.path.getsize(args.out)/1e6:.0f} MB, "
          f"step {ck['step']})", flush=True)


if __name__ == "__main__":
    main()
