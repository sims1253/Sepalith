#!/usr/bin/env python3
"""Fail-closed primitives for full Trainer-state transport."""
from __future__ import annotations
import hashlib,json,os,re,stat,tempfile
from pathlib import Path,PurePosixPath
REQUIRED={'model.safetensors','optimizer.pt','scheduler.pt','rng_state.pth','trainer_state.json','training_args.bin','campaign-state.json','config.json','generation_config.json','tokenizer.json','tokenizer_config.json','chat_template.jinja'}
PAYLOAD_REQUIRED=REQUIRED|{'campaign-manifest.json'}
SECRET=re.compile(r'hf_[A-Za-z0-9]+')
def require(ok,msg):
 if not ok:raise ValueError(msg)
def safe_message(value):
 text=SECRET.sub('[REDACTED]',str(value));text=re.sub(r'(?i)(bearer\s+)[A-Za-z0-9._~+/=-]+',r'\1[REDACTED]',text);text=re.sub(r'(?i)(https?://[^?\s]+)\?\S+',r'\1?[REDACTED]',text);return text[-1024:]
def sha256(path,chunk=8<<20):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(chunk),b''):h.update(b)
 return h.hexdigest()
def hashes(path):
 p=Path(path);size=p.stat().st_size;sha=hashlib.sha256();git=hashlib.sha1(b'blob '+str(size).encode()+b'\0')
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):sha.update(b);git.update(b)
 return {'bytes':size,'sha256':sha.hexdigest(),'git_blob_sha1':git.hexdigest()}
def atomic_json(path,value):
 p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);fd,tmp=tempfile.mkstemp(prefix='.'+p.name+'.',dir=p.parent)
 try:
  with os.fdopen(fd,'w') as f:json.dump(value,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
  os.replace(tmp,p);d=os.open(p.parent,os.O_RDONLY|os.O_DIRECTORY);os.fsync(d);os.close(d)
 finally:
  if os.path.exists(tmp):os.unlink(tmp)
def safe_relative(name):
 q=PurePosixPath(name);return bool(name and not q.is_absolute() and len(q.parts)==1 and '..' not in q.parts and '\\' not in name and '\0' not in name)
def safe_nested(name):
 q=PurePosixPath(name);return bool(name and not q.is_absolute() and q.parts and all(x not in {'','.','..'} for x in q.parts) and '\\' not in name and '\0' not in name)
def load_checkpoint_manifest(checkpoint,expected_sha):
 root=Path(checkpoint);m=root/'campaign-manifest.json';require(m.is_file(),'campaign manifest absent');require(sha256(m)==expected_sha,'campaign manifest pin differs');x=json.loads(m.read_text());require(x.get('schema_version')==1 and x.get('full') is True and x.get('checkpoint_kind')=='full_weights' and x.get('step')==90,'checkpoint identity differs');files=x.get('files');require(isinstance(files,dict) and set(files)==REQUIRED,'full-state file set differs')
 before={}
 for name,row in files.items():
  require(safe_relative(name),'unsafe checkpoint filename');p=root/name;require(p.is_file() and not p.is_symlink(),'missing/nonregular checkpoint file:'+name);s=p.stat();require(type(row.get('bytes')) is int and row['bytes']>0 and s.st_size==row['bytes'],'checkpoint file size differs:'+name);require(isinstance(row.get('sha256'),str) and len(row['sha256'])==64,'checkpoint file hash metadata invalid:'+name);before[name]=(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,stat.S_IMODE(s.st_mode))
 s=m.stat();before['campaign-manifest.json']=(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,stat.S_IMODE(s.st_mode))
 return x,before
def validate_small_state(checkpoint,manifest):
 root=Path(checkpoint);state=json.loads((root/'campaign-state.json').read_text());trainer=json.loads((root/'trainer_state.json').read_text());require(state.get('full') is True and state.get('checkpoint_kind')=='full_weights' and state.get('step')==manifest['step'] and state.get('identity')==manifest['identity'],'campaign state differs');sampler=state.get('sampler',{});require(sampler.get('method')=='sequential_frozen_draw_schedule_stage_local' and sampler.get('cursor')==sampler.get('stage_cursor')==384 and sampler.get('global_optimizer_step_offset')==66 and sampler.get('global_step')==90 and sampler.get('effective_batch')==16,'sampler/cursor identity differs');require(trainer.get('global_step')==90 and trainer.get('max_steps')==11509 and trainer.get('train_batch_size')==1,'trainer cursor differs');return {'step':90,'stage_cursor':384,'global_optimizer_step_offset':66,'effective_batch':16,'draw_schedule_sha256':sampler['draw_schedule_sha256'],'trainer_max_steps':trainer['max_steps']}
def assert_stats_unchanged(checkpoint,snapshot):
 root=Path(checkpoint)
 for name,old in snapshot.items():
  s=(root/name).stat();require((s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,stat.S_IMODE(s.st_mode))==old,'checkpoint changed during transport:'+name)
