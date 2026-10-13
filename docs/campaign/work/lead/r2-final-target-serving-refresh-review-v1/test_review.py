#!/usr/bin/env python3
"""CPU-only executable review controls for the frozen serving-refresh packet."""
from __future__ import annotations
import hashlib,json,os,sys,tempfile,unittest
from pathlib import Path
os.environ.setdefault('OMP_NUM_THREADS','2');os.environ.setdefault('MKL_NUM_THREADS','2')
PLAN=Path('/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb')
FROZEN=PLAN/'docs/campaign/work/lead/r2-final-target-serving-refresh-v1'
LLAMA=Path('/home/m0hawk/Documents/Sepalith/experiments/bin/src/llamacpp-b10453')
sys.path[:0]=[str(FROZEN),str(LLAMA),str(LLAMA/'gguf-py')]
import numpy as np
import torch
import gguf
from conversion.base import ModelBase
import analyze_refresh as analyzer
import prepare_refresh as prepare

def sha(p:Path)->str:
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
class Synthetic(ModelBase):pass
class EmptyMap:mapping={}
class CaptureWriter:
 def __init__(self):self.items=[]
 def add_tensor(self,name,data,raw_dtype=None):self.items.append((name,data.copy(),raw_dtype))
def actual_convert(name:str,tensor:torch.Tensor):
 with tempfile.TemporaryDirectory() as td:
  model=object.__new__(Synthetic);model.hparams={};model.dir_model=Path(td);model.model_tensors={};model.tensor_map=EmptyMap();model.ftype=gguf.LlamaFileType.MOSTLY_F16;model.gguf_writer=CaptureWriter()
  model.dequant_model=lambda:None;model.generate_extra_tensors=lambda:[];model.get_tensors=lambda:[('synthetic',tensor)];model.modify_tensors=lambda data,_name,_bid:[(name,data)];model.tensor_force_quant=lambda *_:False;model.match_model_tensor_name=lambda *_:False
  model.prepare_tensors();return model.gguf_writer.items[0]
def rec(wall=10,ids=(2,1),raw='x',cap=192):
 return {'row_id':'only-row','phase':'cold','rep':1,'protocol_status':'accepted','returned_token_ids':list(ids),'raw_text':raw,'combined_case_wall_ms':wall,'returned_token_count':len(ids),'cap':cap,'draft_n':10,'draft_n_accepted':4}
class ReviewTests(unittest.TestCase):
 def test_pinned_converter_preserves_fp32_norm_exactly(self):
  self.assertEqual(sha(LLAMA/'convert_hf_to_gguf.py'),'e38975e1c68d98ac1664dfd530616eb35c72294382a4dd873d4746b23f27779f')
  source=torch.tensor([1.0000001192092896,-0.3333333432674408],dtype=torch.float32)
  name,data,qtype=actual_convert('blk.0.attn_norm.weight',source)
  self.assertEqual(name,'blk.0.attn_norm.weight');self.assertEqual(qtype,gguf.GGMLQuantizationType.F32);self.assertEqual(data.dtype,np.float32);self.assertTrue(np.array_equal(data,source.numpy()))
 def test_same_converter_casts_ordinary_fp32_matrix_under_f16_outtype(self):
  source=torch.tensor([[1.0000001192092896,-0.3333333432674408]],dtype=torch.float32)
  _,data,qtype=actual_convert('blk.0.attn_q.weight',source)
  self.assertEqual(qtype,gguf.GGMLQuantizationType.F16);self.assertEqual(data.dtype,np.float16);self.assertFalse(np.array_equal(data.astype(np.float32),source.numpy()))
 def test_frozen_analyzer_can_pass_one_request(self):
  bind={'target_identity':'T','tokenizer_json_sha256':'a'*64};base={'artifact_binding':bind,'requests':[rec(10)]};cand={'artifact_binding':bind,'requests':[rec(5)]}
  result=analyzer.compare(base,cand,'candidate');self.assertEqual(result['status'],'pass');self.assertEqual(result['baseline']['requests'],1)
 def test_frozen_cap_heuristic_mislabels_eos_exactly_at_cap(self):
  row=rec(ids=(7,1),cap=2);row['final']={'stop':True,'stop_type':'eos'};row['protocol_checks']={'stop':'eos'}
  self.assertEqual(analyzer.summary({'requests':[row]})['cap_hits'],1)
 def test_frozen_prepare_accepts_absent_hf_weights_and_opaque_manifests(self):
  with tempfile.TemporaryDirectory() as td:
   d=Path(td);hf=d/'hf';hf.mkdir();(hf/'tokenizer.json').write_text('tok');(hf/'config.json').write_text('cfg')
   def item(name,text=None):p=d/name;p.write_text(name if text is None else text);return {'path':str(p),'sha256':sha(p)}
   target_manifest=item('target-manifest','{}');weights_manifest=item('weights-manifest','opaque');selection=item('selection','{}')
   tools={k:item(k) for k in ('convert_hf_to_gguf','imatrix','quantize','server','cuda_backend','panel_probe')};cal=item('cal');calm=item('calm','opaque');panel=item('panel');panelm=item('panelm','opaque')
   b={'schema_version':'sepalith.r2.final-target-serving-refresh.v1','launch_authorized':True,'target':{'identity':'T','hf_dir':str(hf),'manifest_path':target_manifest['path'],'manifest_sha256':target_manifest['sha256'],'tokenizer_json_sha256':sha(hf/'tokenizer.json'),'config_sha256':sha(hf/'config.json'),'weights_manifest_path':weights_manifest['path'],'weights_manifest_sha256':weights_manifest['sha256'],'selection_receipt_path':selection['path'],'selection_receipt_sha256':selection['sha256'],'checkpoint_kind':'full_weights'},'tokenizer_contract':{'tokenizer_json_sha256':sha(hf/'tokenizer.json')},'calibration':{'split':'train','contains_dev_or_final':False,'text_path':cal['path'],'text_sha256':cal['sha256'],'manifest_path':calm['path'],'manifest_sha256':calm['sha256']},'quality_panel':{'split':'train','contains_dev_or_final':False,'path':panel['path'],'sha256':panel['sha256'],'manifest_path':panelm['path'],'manifest_sha256':panelm['sha256']},'tools':tools,'outputs':{'root':str(d/'out'),'f16':{'path':str(d/'fresh.gguf')}},'drafts':{},'runtime':{}}
   commands=prepare.validate_and_commands(b,'export');self.assertTrue(commands);self.assertFalse((hf/'model.safetensors').exists())
if __name__=='__main__':unittest.main(verbosity=2)
