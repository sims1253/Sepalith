import sys, json, hashlib, os
from pathlib import Path
R=Path(__file__).resolve().parent
sys.path.insert(0,str(R))
import supervise_recovery as s
from cache_maintenance import start
sys.path.insert(0,str(s.SOURCE))
from native_checkpoint_publish import publish
stop, worker, errors=start(s.native,s.archive,s.data)
cp=s.native/'checkpoint-4866';dest=s.archive/'full/checkpoint-4866'
m=s.read(cp/'campaign-manifest.json');state=s.read(cp/'campaign-state.json')
assert m['step']==4866 and m['identity']==s.read(s.native/'checkpoint-4802/campaign-manifest.json')['identity']
assert state['sampler']['cursor']==76800 and state['sampler']['global_step']==4866
assert s.read(cp/'trainer_state.json')['global_step']==4866
receipt=publish(cp,dest,s.sha(cp/'campaign-manifest.json'))
for name,expected in m['files'].items():
 p=dest/name
 assert p.stat().st_size==expected['bytes']
 assert s.sha(p)==expected['sha256'],name
assert not errors,errors
s.write(R/'checkpoint-4866-recovered.json',{'status':'native_and_durable_payloads_verified','step':4866,'cursor':76800,'publication':receipt,'cache_policy':'continuous targeted clean page release; guards unchanged'})
print('PASS: checkpoint 4866 full payloads published and independently verified',flush=True)
