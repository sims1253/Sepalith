"""Consume a verify-run inherited file attestation without rereading payloads."""
import json,os
from pathlib import Path

def require(v,m):
 if not v:raise ValueError(m)
def fp(s):return {k:int(getattr(s,k)) for k in ('st_dev','st_ino','st_size','st_mtime_ns','st_ctime_ns')}

def load_attestation(expected_bundle_id):
 raw=os.environ.get('SEPALITH_NATIVE_STAGE_ATTESTATION');require(raw,'native verifier attestation missing')
 value=json.loads(raw);require(value.get('bundle_id')==expected_bundle_id,'native bundle identity differs')
 records=value.get('files');require(isinstance(records,list) and records,'native attested files empty')
 by_path={item['path']:item for item in records};require(len(by_path)==len(records),'native attested paths duplicate')
 return value,by_path

def require_attested(path,expected_sha256,expected_bytes,bundle_id):
 path=Path(path).resolve();_,records=load_attestation(bundle_id);record=records.get(str(path));require(record is not None,'file lacks native verifier attestation')
 require(record.get('sha256')==expected_sha256 and record.get('bytes')==expected_bytes,'attested content identity differs')
 fd=record.get('fd');require(type(fd)is int and fd>=3,'attested descriptor differs')
 held=os.fstat(fd);current=path.stat();expected=record.get('fingerprint')
 require(fp(held)==expected and fp(current)==expected,'attested file identity changed after verification')
 require(Path(f'/proc/self/fd/{fd}').resolve()==path,'attested descriptor points to another file')
 return {'path':str(path),'sha256':expected_sha256,'bytes':expected_bytes,'fd':fd,'bundle_id':bundle_id}

def require_attested_identity(path,bundle_id):
 path=Path(path).resolve();_,records=load_attestation(bundle_id);record=records.get(str(path));require(record is not None,'file lacks native verifier attestation')
 return require_attested(path,record['sha256'],record['bytes'],bundle_id)
