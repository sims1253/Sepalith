#!/usr/bin/env python3
from __future__ import annotations
import argparse,hashlib,json,shutil,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
OLD=HERE.parent/'r2-notebook-dev-syntax-v1';sys.path.insert(0,str(OLD))
from prepare_replay import load_protocol,splice
PANEL=HERE.parent/'corrected-dev75-v1/dev75-corrected-finish-v1.jsonl';PANEL_SHA='7c144bcd20570b1b1b1a232a5d90589c281ac2955d5fc2ef4b4b01fed8d57035'
RESULT=Path('/mnt/e/sepalith/campaign-20260915/training/SFT11-expanded-e750-native-dev-v1/dev-results/results.json');RESULT_SHA='aaff0befb2926c0c6ff4b5279300af7426eb9ac69105e64dbe8748db6775fd8c'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def req(x,m):
 if not x:raise ValueError(m)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();req(not a.output.exists(),'fresh output required')
 req(sha(PANEL)==PANEL_SHA and sha(RESULT)==RESULT_SHA,'input hash mismatch');panel=[json.loads(x) for x in PANEL.open()];doc=json.load(RESULT.open());ids=[x['id'] for x in panel]
 req(len(panel)==75 and doc['expected_case_ids']==ids and doc['completed_case_ids']==ids and len(doc['results'])==75,'case identity')
 protocol=load_protocol();a.output.mkdir();buffers=a.output/'buffers';buffers.mkdir();rows=[]
 for i,(p,native) in enumerate(zip(panel,doc['results'])):
  req(native['id']==p['id'] and native['family']==p['family'] and native['prompt_sha256']==p['prompt_sha256'],'native identity')
  cdir=buffers/f'{i:03d}';cdir.mkdir();oldbase=OLD/'packet/buffers'/f'{i:03d}/baseline.R';baseline=oldbase.read_bytes();req(hashlib.sha256(baseline).hexdigest()==p['context']['replacement_range']['content_sha256'],'reviewed baseline identity')
  bpath=cdir/'baseline.R';bpath.write_bytes(baseline);context=protocol.PromptContext.from_dict(p['context']);parsed=protocol.parse_output(native['raw_text'],context)
  req(parsed.status==native['protocol']['parser_status'] and parsed.operation==native['protocol']['parser_operation'] and ((parsed.status=='accepted')==native['protocol']['valid']),'context parser mismatch')
  apath=None;asha=None
  if parsed.status=='accepted':
   if parsed.operation=='no_op':applied=baseline
   else:
    eol='\r\n' if context.document_eol=='crlf' else '\n';applied=splice(baseline.decode(),p['context']['replacement_range'],parsed.body_text,eol).encode()
   apath=cdir/'E750.R';apath.write_bytes(applied);asha=hashlib.sha256(applied).hexdigest()
  rows.append({'arm':'E750','case_index':i,'id':p['id'],'family':p['family'],'expected_operation':p['operation'],'prompt_sha256':p['prompt_sha256'],
   'baseline_relative_path':str(bpath.relative_to(a.output)),'baseline_sha256':hashlib.sha256(baseline).hexdigest(),'protocol_valid':parsed.status=='accepted','protocol_status':parsed.status,
   'protocol_operation':parsed.operation,'protocol_reason':parsed.reason,'raw_text_sha256':native['raw_text_sha256'],'applied_relative_path':str(apath.relative_to(a.output)) if apath else None,
   'applied_sha256':asha,'edit_exact':native['quality']['edit_exact'],'strict_noop_correct':native['quality']['strict_noop_correct'],'noop_false_positive':native['quality']['noop_false_positive']})
 with (a.output/'replay-inputs.jsonl').open('w') as f:
  for x in rows:f.write(json.dumps(x,sort_keys=True)+'\n')
 (a.output/'manifest.json').write_text(json.dumps({'schema':'sepalith.notebook-e750-dev-syntax-input.v1','panel_sha256':PANEL_SHA,'result_sha256':RESULT_SHA,'rows':75,'protocol_valid':sum(x['protocol_valid'] for x in rows),'protocol_invalid':sum(not x['protocol_valid'] for x in rows),'old_reviewed_packet_manifest_sha256':sha(OLD/'packet/manifest.json'),'raw_wire_parsed_context_aware':True,'decoded_body_text_used':False,'generated_r_executed':False},indent=2,sort_keys=True)+'\n')
if __name__=='__main__':main()
