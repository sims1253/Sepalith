#!/usr/bin/env python3
import hashlib,json,re
from pathlib import Path
PLAN=Path('/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb')
UNS=Path('/home/m0hawk/Documents/Sepalith/.venv-sft/lib/python3.10/site-packages/unsloth')
XF=Path('/mnt/e/sepalith/campaign-20260915/build-work/sm120-varlen-v1/cu130-overlay/xformers')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def need(v,m):
 if not v:raise AssertionError(m)
pins={
 UNS/'utils/attention_dispatch.py':'9f0b160da7e3105586c252a9f6384218ce00736ecca2b0a72b741412934b5a31',
 UNS/'models/llama.py':'8a79d94ab5f7f76c241d49da9bb1af9a90a9491977d3d08bbf356334dd007f2d',
 Path('/home/m0hawk/Documents/Sepalith/.venv-sft/lib/python3.10/site-packages/torch/nn/attention/varlen.py'):'9efc95d28ad93a8567ca695cd4b00875692a01e2a91f140aecdb60feb05cca00',
 XF/'ops/fmha/dispatch.py':'b99fd512a242b7f48d916646b8c6b47dde587afa2d7884580cc69f9f8ce6d9ae',
 XF/'ops/fmha/flash3.py':'a86c0f009ac0ba3fdfbfe6164983ba3b62959a753877f459a1836f7f65be2a8b',
 XF/'ops/fmha/cutlass.py':'aa8a5ddd43879732d13ea6a19a94b7cf2fcca809f4abc465db5cfe98637437a5',
 XF/'_C.so':'bb2a59af5ed03aa28ea0e1ed705384fcb6d5d0e6ab8bd825b004ffbb3af9d669'}
for p,h in pins.items():need(sha(p)==h,'source pin differs:'+str(p))
d=(XF/'ops/fmha/dispatch.py').read_text();need(d.index('flash3.FwOp')<d.index('flash.FwOp')<d.index('cutlass.FwOp'),'forward priority differs');need('cutlass_blackwell' not in d,'blackwell operator entered auto dispatcher')
need('CUDA_MAXIMUM_COMPUTE_CAPABILITY = (9, 0)' in (XF/'ops/fmha/flash3.py').read_text(),'FA3 cap differs');need('CUDA_MAXIMUM_COMPUTE_CAPABILITY = (9, 0)' in (XF/'ops/fmha/cutlass.py').read_text(),'CUTLASS cap differs')
a=(UNS/'utils/attention_dispatch.py').read_text();need('K_mod = K_mod.reshape(bsz, kv_seq_len, n_heads, head_dim)' in a and 'V_mod = V_mod.reshape(bsz, kv_seq_len, n_heads, head_dim)' in a,'training GQA materialization differs')
t=Path('/home/m0hawk/Documents/Sepalith/.venv-sft/lib/python3.10/site-packages/torch/nn/attention/varlen.py').read_text();need('return False' in t[t.index('def _should_use_cudnn'):t.index('class AuxRequest')],'cuDNN selection differs');need('aten._flash_attention_forward' in t and 'aten._flash_attention_backward' in t and '_varlen_attn.register_autograd' in t,'Torch varlen autograd path differs')
cfg=json.loads(Path('/home/m0hawk/.local/state/sepalith/campaign-20260915/training/SFT11-native-CPT-prefix-extension-v3-from194/runtime/checkpoint-322/config.json').read_text());need((cfg['num_hidden_layers'],cfg['num_attention_heads'],cfg['num_key_value_heads'],cfg['head_dim'])==(42,16,2,128),'model attention layout differs')
paired=PLAN/'docs/campaign/work/lead/r2-native-varlen322-packed-retry-root-v1/paired-intermediate.json';need(sha(paired)=='0fa384314483bb7d2c59baa18d3f9f2549da830d3d3f93028c77afd39711a6c0','paired result differs');x=json.loads(paired.read_text());need(x['ordinary_over_varlen_speedup']<1 and x['maximum_relative_loss_difference']<=x['maximum_allowed_relative_loss_difference'],'paired decision differs')
print('PASS pinned dispatch priority/caps, GQA copy, Torch varlen autograd, model layout and paired result')
