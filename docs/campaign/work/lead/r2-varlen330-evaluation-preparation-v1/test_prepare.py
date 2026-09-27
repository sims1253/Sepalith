#!/usr/bin/env python3
import copy,hashlib,importlib.util,json,tempfile
from pathlib import Path
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('prep330',HERE/'prepare.py');P=importlib.util.module_from_spec(spec);spec.loader.exec_module(P)
identity={'arm':'ordinary_reference'}
with tempfile.TemporaryDirectory(prefix='varlen330-test-') as td:
 root=Path(td);native=root/'native';durable=root/'durable';native.mkdir();durable.mkdir();files={}
 for name in sorted(P.FILES):
  if name=='campaign-state.json':data=json.dumps({'identity':identity,'step':330,'sampler':{'cursor':4224,'stage_cursor':4224,'global_step':330,'global_optimizer_step_offset':66,'effective_batch':16}}).encode()
  elif name=='trainer_state.json':data=json.dumps({'global_step':330}).encode()
  else:data=(name+'\n').encode()
  files[name]={'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}
  (native/name).write_bytes(data);(durable/name).write_bytes(data)
 manifest={'full':True,'checkpoint_kind':'full_weights','step':330,'identity':identity,'files':files}
 for d in (native,durable):(d/'campaign-manifest.json').write_text(json.dumps(manifest,sort_keys=True)+'\n')
 assert P.verify_pair(native,durable,identity)==manifest
 (durable/'optimizer.pt').write_bytes(b'bad')
 try:P.verify_pair(native,durable,identity)
 except ValueError:pass
 else:raise AssertionError('corrupt optimizer accepted')
recipe={'parent':{'files':{'tokenizer.json':'tok'}},'cohort':{'rows':{'sha256':'rows'},'streaming_cache':{'manifest_sha256':'cache'},'draw_schedule':{'sha256':'schedule'}},'runtime':{'optimizer':{'arm':'aurora_mix'}},'varlen_canary':{'arm':'ordinary_reference'}}
sci={'tokenizer_sha256':'tok','rows_sha256':'rows','cache_manifest_sha256':'cache','schedule_sha256':'schedule','optimizer_sha256':hashlib.sha256(json.dumps(recipe['runtime']['optimizer'],sort_keys=True,separators=(',',':')).encode()).hexdigest()}
report={'status':'root_admitted_execution_stopped','canary_arm':'ordinary_reference','terminal_checkpoint':str(Path('/durable')),'source_global_step':322,'global_step':330,'global_optimizer_step_offset':66,'initial_cursor':4096,'observed_draws':128,'last_draw_position':4223,'logical_updates':8,'logical_rows':128,'source_checkpoint_manifest_sha256':'81dfdfb4af6d0690ebb114f68910fdbe270d20ad3393a84c47c1c337ce2dbbcf','scientific_bindings':sci,'canary_admission':{'path':str((P.CANARY/'ordinary_reference/admission.json').resolve()),'sha256':P.PINS['ordinary_reference']['admission'],'arm':'ordinary_reference','source_step':322,'target_step':330,'source_cursor':4096,'target_cursor':4224},'execution_stop_admission':{'stop_at_global_step':330},'logical_update_losses':[1.0]*8,'loss_denominators':[10]*8,'warmup_update_seconds':[1.0]*2,'timed_update_seconds':[1.0]*6,'attention_backend':'sdpa_ordinary','loss_reduction':'global_supervised_token_mean','draw_ids_sha256':'draws'}
assert P.validate_report(report,recipe,'ordinary_reference',Path('/durable'))['draw_ids_sha256']=='draws'
for key,value in [('canary_arm','varlen_candidate'),('initial_cursor',4095),('observed_draws',127)]:
 bad=copy.deepcopy(report);bad[key]=value
 try:P.validate_report(bad,recipe,'ordinary_reference',Path('/durable'))
 except ValueError:pass
 else:raise AssertionError(key+' mutation accepted')
print('PASS synthetic full12 native/durable payload, corruption, arm, cursor and draw-count controls')
