"""Compare the real upstream draft's meta-device state to public tensor headers."""
import json, sys, struct, hashlib, datetime
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'vendor/DeepSpec'))
import torch
from transformers import AutoConfig
from deepspec.modeling.dspark.qwen3.config import build_draft_config
from deepspec.modeling.dspark.qwen3.modeling import Qwen3DSparkModel
from deepspec.utils.config import ConfigNode
N=Path('/home/m0hawk/.local/state/sepalith/campaign-20260915')
target=AutoConfig.from_pretrained(N/'models/SFT11-CPT-global-a-250-merged',local_files_only=True)
args=ConfigNode(dict(num_draft_layers=5,target_layer_ids=[1,10,20,30,39],confidence_head_alpha=1.,confidence_head_with_markov=True,markov_rank=256,markov_head_type='vanilla',block_size=7,mask_token_id=75982,num_anchors=32))
config=build_draft_config(target,args);config.sliding_window=getattr(target,'sliding_window',None)
with torch.device('meta'):
 model=Qwen3DSparkModel(config)
state=model.state_dict();assert all(v.device.type=='meta' for v in state.values())
p=N/'models/released-dspark-hf-v1/model.safetensors'
with p.open('rb') as f:
 size=struct.unpack('<Q',f.read(8))[0];assert size<100000;raw=f.read(size)
header=json.loads(raw);tensors={k:v for k,v in header.items() if k!='__metadata__'}
missing=set(state)-set(tensors);extra=set(tensors)-set(state)
assert missing=={'embed_tokens.weight','lm_head.weight'},missing
assert not extra,extra
for key,v in tensors.items():
 assert list(state[key].shape)==v['shape'],key
 assert v['dtype']=='BF16',key
r={'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'task':'RUN-08','status':'actual_full_geometry_meta_header_match','upstream_revision':'005e03b81cec38b7da6399833d609ee89a2587f2','class':type(model).__module__+'.'+type(model).__name__,'model_state_tensors':len(state),'public_tensors':len(tensors),'exact_missing_keys':sorted(missing),'all_present_shapes_match':True,'public_dtype':'BF16','meta_device_only':True,'model_weights_loaded':False,'cuda_started':False,'header_sha256':hashlib.sha256(raw).hexdigest(),'target_config_sha256':hashlib.sha256((N/'models/SFT11-CPT-global-a-250-merged/config.json').read_bytes()).hexdigest(),'num_anchors_profile_candidate':32,'acceptance_limit':'Shape compatibility only; actual GPU forward/backward and frozen target head initialization remain required'}
(ROOT/'meta-warmstart-gate.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
