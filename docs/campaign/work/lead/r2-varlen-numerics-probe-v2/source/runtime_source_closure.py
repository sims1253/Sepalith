"""Strict source-closure verification for a runtime entrypoint."""
from __future__ import annotations
import hashlib,json
from pathlib import Path

def sha256(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def verify_runtime_source(manifest_path,expected_schema):
 p=Path(manifest_path);root=p.parent.resolve();m=json.loads(p.read_text())
 if m.get('schema')!=expected_schema or m.get('status')!='frozen_preparation':raise ValueError('runtime source manifest schema/status differs')
 files=m.get('files')
 if not isinstance(files,list) or not files:raise ValueError('runtime source manifest is empty')
 seen=set()
 for item in files:
  rel=item.get('path')
  if not isinstance(rel,str) or rel in seen:raise ValueError('runtime source path differs')
  target=(p.parent/rel).resolve()
  if not target.is_relative_to(root):raise ValueError('runtime source escapes packet')
  seen.add(rel)
  if not target.is_file() or target.is_symlink() or target.stat().st_size!=item.get('bytes') or sha256(target)!=item.get('sha256'):raise ValueError(f'runtime source differs:{rel}')
 return {'path':str(p.resolve()),'sha256':sha256(p),'schema':expected_schema,'files':len(files)}
