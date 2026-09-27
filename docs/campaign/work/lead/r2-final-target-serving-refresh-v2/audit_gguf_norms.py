#!/usr/bin/env python3
"""Header plus small-tensor proof that saved FP32 norms survive GGUF stages."""
import argparse,hashlib,json,os,re,tempfile
from pathlib import Path
import numpy as np
from safetensors import safe_open
from prepare_refresh_v2 import validate_binding,artifact,require,write_new
HERE=Path(__file__).resolve().parent
def sha_bytes(x):return hashlib.sha256(memoryview(np.ascontiguousarray(x)).cast('B')).hexdigest()
def mapped(name):
 if name=='model.norm.weight':return 'output_norm.weight'
 m=re.fullmatch(r'model\.layers\.(\d+)\.input_layernorm\.weight',name)
 if m:return f'blk.{m.group(1)}.attn_norm.weight'
 m=re.fullmatch(r'model\.layers\.(\d+)\.post_attention_layernorm\.weight',name)
 if m:return f'blk.{m.group(1)}.ffn_norm.weight'
 raise ValueError(f'unreviewed FP32 norm mapping: {name}')
def audit(binding,key):
 b=json.loads(Path(binding).read_text());validate_binding(b,require_admission=True,check_large_payloads=True);target=Path(b['target']['hf_dir'])/'model.safetensors';gguf_path=artifact(b['outputs'][key],key)
 expected={}
 with safe_open(target,framework='pt',device='cpu') as f:
  saved_fp32=0
  for name in f.keys():
   dtype=f.get_slice(name).get_dtype()
   if dtype=='F32' and not name.endswith('norm.weight'):raise ValueError(f'saved non-norm F32 tensor needs reviewed mapping: {name}')
   if name.endswith('norm.weight'):
    require(dtype in {'F32','BF16','F16'},f'unsupported saved norm dtype: {name} {dtype}');saved_fp32 += dtype=='F32';a=f.get_tensor(name).float().numpy();expected[mapped(name)]=(a,dtype)
 require(len(expected)==b['target']['expected_norm_count'] and saved_fp32==b['target']['expected_fp32_norm_count'],'saved norm dtype counts differ')
 import sys;root=Path(b['converter_source']['root']);sys.path.insert(0,str(root/'gguf-py'));from gguf import GGUFReader
 reader=GGUFReader(gguf_path);actual={t.name:t for t in reader.tensors};require(len(actual)==b['target']['expected_tensor_count'],'GGUF tensor count differs');rows=[]
 for name,(x,saved_dtype) in sorted(expected.items()):
  require(name in actual,f'missing GGUF norm {name}');t=actual[name];require(t.tensor_type.name=='F32',f'GGUF norm cast: {name}');y=np.asarray(t.data,dtype=np.float32).reshape(-1);z=np.asarray(x,dtype=np.float32).reshape(-1);require(y.shape==z.shape and np.array_equal(y,z),f'GGUF norm values differ: {name}');rows.append({'name':name,'saved_dtype':saved_dtype,'values':len(z),'sha256':sha_bytes(z)})
 return {'schema':'sepalith.run06.gguf-saved-norm-audit.v1','status':'pass','target_identity':b['target']['identity'],'artifact_key':key,'artifact_sha256':b['outputs'][key]['sha256'],'saved_weights_sha256':b['target']['weights_sha256'],'norms':len(rows),'saved_fp32_norms':saved_fp32,'rows':rows,'bit_exact':True}
def main():
 p=argparse.ArgumentParser();p.add_argument('--binding',required=True);p.add_argument('--artifact-key',choices=('f16','q8','q4_k_m','iq3_m'),required=True);p.add_argument('--out',required=True);a=p.parse_args();write_new(a.out,audit(a.binding,a.artifact_key));print(json.dumps({'status':'pass','artifact':a.artifact_key}))
if __name__=='__main__':main()
