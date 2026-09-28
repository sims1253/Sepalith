#!/usr/bin/env python3
"""Publish an already sealed native checkpoint to durable E atomically.

Copying and SHA-256 validation happen in one source pass.  The destination is
not visible at its final name until every payload is fsynced and the manifest
has been written in the temporary directory.
"""
from __future__ import annotations
import argparse, hashlib, json, os
from pathlib import Path
import shutil, tempfile

BLOCK = 8 * 1024 * 1024

def require(v, m):
    if not v: raise ValueError(m)

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(BLOCK),b''):h.update(b)
    return h.hexdigest()

def flush_dir(path):
    fd=os.open(path,os.O_RDONLY|os.O_DIRECTORY)
    try: os.fsync(fd)
    except OSError as e:
        if e.errno not in (22,95): raise
    finally: os.close(fd)

def fp(s): return tuple(int(getattr(s,k)) for k in ('st_dev','st_ino','st_size','st_mtime_ns','st_ctime_ns'))

def copy_hash(source,destination,expected):
    source=Path(source);destination=Path(destination)
    source_fd=os.open(source,os.O_RDONLY|getattr(os,'O_NOFOLLOW',0));before=os.fstat(source_fd)
    require(os.path.isfile(f'/proc/self/fd/{source_fd}'),'checkpoint payload is not regular')
    destination.parent.mkdir(parents=True,exist_ok=True)
    h=hashlib.sha256();total=0
    try:
      with os.fdopen(os.dup(source_fd),'rb') as src,destination.open('xb') as out:
          for block in iter(lambda:src.read(BLOCK),b''):
              h.update(block);out.write(block);total+=len(block)
          out.flush();os.fsync(out.fileno())
      require(fp(before)==fp(os.fstat(source_fd)),f'source changed during publication:{source.name}')
    finally:os.close(source_fd)
    require(total==expected['bytes'] and h.hexdigest()==expected['sha256'],f'source bytes differ:{source.name}')
    os.chmod(destination,0o400)
    return {'bytes':total,'sha256':h.hexdigest(),'source_fingerprint':fp(before)}

def publish(source,destination,expected_manifest_sha256,*,require_cross_filesystem=True):
    source=Path(source);destination=Path(destination)
    require(source.is_dir() and not source.is_symlink(),'sealed native checkpoint missing')
    manifest_path=source/'campaign-manifest.json'
    manifest_bytes=manifest_path.read_bytes()
    require(hashlib.sha256(manifest_bytes).hexdigest()==expected_manifest_sha256,'sealed native manifest differs')
    manifest=json.loads(manifest_bytes)
    require(manifest.get('checkpoint_kind')=='full_weights' and manifest.get('full') is True,'checkpoint is not full resume state')
    required={'model.safetensors','optimizer.pt','scheduler.pt','rng_state.pth','trainer_state.json','campaign-state.json','tokenizer.json'}
    require(required<=set(manifest.get('files',{})),'full checkpoint inventory is incomplete')
    actual=set()
    for path in source.rglob('*'):
        require(not path.is_symlink(),f'checkpoint contains symlink:{path}')
        if path.is_file():actual.add(str(path.relative_to(source)))
    require(actual==set(manifest['files'])|{'campaign-manifest.json'},'checkpoint file set differs from sealed manifest')
    destination.parent.mkdir(parents=True,exist_ok=True)
    require(not destination.exists(),'durable destination already exists')
    if require_cross_filesystem: require(destination.parent.stat().st_dev != source.stat().st_dev,'publisher is for native-to-durable cross-filesystem copy')
    temporary=Path(tempfile.mkdtemp(prefix=f'.{destination.name}.',dir=destination.parent));copied={}
    try:
        for name,expected in sorted(manifest['files'].items()):
            rel=Path(name);require(not rel.is_absolute() and '..' not in rel.parts,'manifest path escapes checkpoint')
            copied[name]=copy_hash(source/rel,temporary/rel,expected)
        target_manifest=temporary/'campaign-manifest.json'
        with target_manifest.open('xb') as out:
            out.write(manifest_bytes);out.flush();os.fsync(out.fileno())
        require(digest(target_manifest)==expected_manifest_sha256,'copied manifest differs')
        flush_dir(temporary);os.rename(temporary,destination);flush_dir(destination.parent)
        require(json.loads((destination/'campaign-manifest.json').read_text())==manifest,'published manifest differs')
        receipt={'schema':'sepalith.sft11.native-checkpoint-publish.v1','status':'durable_atomic_complete','source':str(source.resolve()),'destination':str(destination.resolve()),'manifest_sha256':expected_manifest_sha256,'step':manifest['step'],'files':copied}
        return receipt
    finally:
        if temporary.exists():shutil.rmtree(temporary)

def main():
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--destination',required=True);p.add_argument('--manifest-sha256',required=True);p.add_argument('--receipt',required=True);a=p.parse_args()
    value=publish(a.source,a.destination,a.manifest_sha256)
    receipt=Path(a.receipt);receipt.parent.mkdir(parents=True,exist_ok=True)
    with receipt.open('x') as out:json.dump(value,out,indent=2,sort_keys=True);out.write('\n');out.flush();os.fsync(out.fileno())
    print(json.dumps(value,sort_keys=True));return 0

if __name__=='__main__':raise SystemExit(main())
