#!/usr/bin/env python3
"""Bind a root-owned imatrix terminal envelope to exact inputs and output."""
import argparse, hashlib, json
from pathlib import Path
from prepare_refresh_v3 import artifact, imatrix_argv, require, sha, validate_binding, write_new

def command_sha(argv):
 return hashlib.sha256(json.dumps(argv,separators=(',',':')).encode()).hexdigest()

def record(binding_path, execution_path, out):
 b=json.loads(Path(binding_path).read_text());validate_binding(b,True,True)
 execution_path=Path(execution_path);x=json.loads(execution_path.read_text());expected=imatrix_argv(b)
 require(x.get('schema')=='sepalith.run06.root-command-terminal.v1' and x.get('status')=='completed','imatrix terminal envelope differs')
 require(x.get('argv')==expected and x.get('command_sha256')==command_sha(expected),'imatrix command identity differs')
 require(x.get('exit_code')==0,'imatrix exit was not zero');require(x.get('processed_chunks')==b['calibration']['chunks']==64,'imatrix processed chunk count differs');require(x.get('non_embedding_coverage_mismatches')==0,'imatrix non-embedding coverage differs')
 im=artifact(b['outputs']['imatrix'],'imatrix')
 value={'schema':'sepalith.run06.imatrix-completion.v1','status':'pass','binding_sha256':sha(binding_path),'execution_path':str(execution_path.resolve()),'execution_sha256':sha(execution_path),'exit_code':0,'processed_chunks':64,'non_embedding_coverage_mismatches':0,'artifact_sha256':sha(im),'f16_sha256':b['outputs']['f16']['sha256'],'calibration_sha256':b['calibration']['text_sha256'],'tool_sha256':b['tools']['imatrix']['sha256']}
 write_new(out,value);return value

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--binding',required=True);p.add_argument('--execution',required=True);p.add_argument('--out',required=True);a=p.parse_args();print(json.dumps(record(a.binding,a.execution,a.out),sort_keys=True))
