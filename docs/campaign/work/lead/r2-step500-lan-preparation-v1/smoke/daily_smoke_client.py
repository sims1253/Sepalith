#!/usr/bin/env python3
"""Two bounded TRAIN fixture requests via the notebook's daily LAN forward."""
import hashlib,json,sys,time,urllib.request
from pathlib import Path
HERE=Path(__file__).resolve().parent
expected=sys.argv[1]
fixture=HERE/'train-fixture.jsonl'
assert hashlib.sha256(fixture.read_bytes()).hexdigest()==sys.argv[2]
row=json.loads(fixture.read_text().splitlines()[0]);assert row['split']=='train'
binding=json.loads(Path('/home/m0hawk/.local/share/sepalith-r2-step500-checks/current-binding.json').read_text());assert binding['instanceId']==expected
records=[]
def call(route,body=None):
 data=None if body is None else json.dumps(body,separators=(',',':')).encode()
 req=urllib.request.Request(binding['endpoint']+route,data=data,headers={'Content-Type':'application/json'})
 start=time.monotonic()
 with urllib.request.urlopen(req,timeout=5) as response:
  assert response.status==200
  assert response.headers['X-Sepalith-Instance-Id']==expected
  raw=response.read(1048577);assert len(raw)<=1048576
  record={'route':route,'request_id':response.headers.get('X-Sepalith-Request-Id'),'instance_id':expected,'request_sha256':hashlib.sha256(data or b'').hexdigest(),'response_sha256':hashlib.sha256(raw).hexdigest(),'wall_ms':1000*(time.monotonic()-start)}
  records.append(record)
  return raw
runtime=json.loads(call('/sepalith/runtime'));assert runtime['instanceId']==expected
native=json.loads(call('/tokenize',{'content':row['prompt_text'],'add_special':False,'parse_special':False}))['tokens']
assert native==row['input_ids'][1:row['target_start']]
outputs=[]
for warm in (False,True):
 raw=call('/completion',{'prompt':[0,*native],'n_predict':192,'temperature':0,'stream':True,'cache_prompt':warm,'return_tokens':True})
 chunks=[json.loads(line[6:]) for line in raw.decode().splitlines() if line.startswith('data: ') and line[6:]!='[DONE]']
 final=[x for x in chunks if x.get('stop')];assert len(final)==1
 ids=[i for x in chunks for i in x.get('tokens',[])];text=''.join(x.get('content','') for x in chunks)
 assert ids and ids[-1]==1 and len(ids)<=192
 assert final[0]['stop_type']=='eos' and not final[0]['truncated']
 assert final[0]['tokens_evaluated']==len(native)+1
 assert text.endswith('>>>>>>> UPDATED')
 outputs.append({'warm':warm,'token_ids':ids,'raw_text':text,'raw_sha256':hashlib.sha256(text.encode()).hexdigest(),'prompt_tokens':len(native)+1,'timings':final[0].get('timings'),'stop_type':final[0]['stop_type']})
assert outputs[0]['token_ids']==outputs[1]['token_ids']
assert json.loads(call('/sepalith/runtime'))==runtime
print(json.dumps({'status':'complete','split':'train','fixture_id':row['id'],'instance_id':expected,'requests':records,'outputs':outputs,'scope':'Notebook HTTP lifecycle smoke; timings include SSH-forwarded HTTP but exclude editor/provider work. No final quality or percentile claim.'}))
