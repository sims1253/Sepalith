"""Root-only completed-GGUF integrity review; no inference/model tensor load."""
import argparse,collections,gc,hashlib,json,sys
from pathlib import Path
from export_candidate import sha

def main():
 a=argparse.ArgumentParser();a.add_argument('--export',type=Path,required=True);a.add_argument('--output',type=Path,required=True);x=a.parse_args()
 assert not x.output.exists();m=json.loads((x.export/'artifacts.json').read_text());assert m['status']=='exported_pending_independent_integrity'
 paths={Path(r['path']).name:r for r in m['files']}
 for r in paths.values():assert sha(Path(r['path']))==r['sha256'] and Path(r['path']).stat().st_size==r['bytes']
 sys.path.insert(0,'/home/m0hawk/Documents/Sepalith/experiments/bin/src/llamacpp-b10453/gguf-py')
 from gguf import GGUFReader
 f=GGUFReader(paths['model-F16.gguf']['path']);shapes={t.name:[int(v) for v in t.shape] for t in f.tensors};meta={k:v.contents() for k,v in f.fields.items() if k.startswith('tokenizer.')};del f;gc.collect()
 q=GGUFReader(paths['model-Q8_0.gguf']['path']);assert len(q.tensors)==381
 assert {t.name:[int(v) for v in t.shape] for t in q.tensors}==shapes
 assert {k:v.contents() for k,v in q.fields.items() if k.startswith('tokenizer.')}==meta
 assert dict(collections.Counter(t.tensor_type.name for t in q.tensors))=={'Q8_0':296,'F32':85}
 assert all(t.tensor_type.name=='Q8_0' for t in q.tensors if t.name in ('output.weight','token_embd.weight'))
 extents=sorted((int(t.data_offset),int(t.n_bytes)) for t in q.tensors);assert all(a+b==c for (a,b),(c,_) in zip(extents,extents[1:]));assert sum(extents[-1])==paths['model-Q8_0.gguf']['bytes']
 result={'status':'Q8_integrity_verified_pending_native_quality','parent_manifest_sha256':m['parent_manifest_sha256'],'f16':paths['model-F16.gguf'],'q8':paths['model-Q8_0.gguf'],'tensor_count':381,'tensor_types':{'Q8_0':296,'F32':85},'tokenizer_metadata_equal_F16':True,'extent_closure':True,'native_tokenizer_EOG_and_DEV_still_required':True}
 x.output.write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':main()
