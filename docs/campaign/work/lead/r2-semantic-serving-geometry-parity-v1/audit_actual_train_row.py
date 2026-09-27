#!/usr/bin/env python3
import hashlib,json
from pathlib import Path
RID='227626e3a7a234b808a096b1'
PACKETS=Path('/mnt/e/sepalith/campaign-20260915/data-work/DAT10-novel-v1/source-walk-shards-v1/shard-0005/structured-materialization-v1/candidate-packets.jsonl')
PACKETS_SHA='948481ce2e9cba79505d5ab20b9ae42812eb945ebe644f60735875a1c169dd9b'
SOURCE=Path('/mnt/h/sepalith/normalized/fastbioclim/0.4.2/fastbioclim/R/bios_fast.R')
SOURCE_SHA='75b80a9d2f56e86a4264f4d786e4c7bbd777bb8b8a42b40a1077ff34232a1c22'
def sha(b):return hashlib.sha256(b).hexdigest()
raw_packets=PACKETS.read_bytes();assert sha(raw_packets)==PACKETS_SHA
packet=next(json.loads(x) for x in raw_packets.splitlines() if json.loads(x).get('row_ref',{}).get('row_id')==RID)
source=SOURCE.read_bytes();assert sha(source)==SOURCE_SHA
result=packet['result'];context=result['context'];selection=result['selection_source'];target=('\n'.join(result['target_body'])+'\n').encode();assert source.count(target)==1
view='\n'.join([*context['prefix'],'',*context['suffix_lines']]).encode();assert view==selection['text'].encode();assert sha(view)==selection['content_sha256']==context['replacement_range']['content_sha256']
global_line=source[:source.index(target)].count(b'\n');local_line=context['replacement_range']['start']['line'];assert global_line!=local_line;assert sha(source)!=sha(view)
out={'schema':'sepalith.run06.actual_train_geometry_mismatch.v1','status':'candidate_geometry_not_current_editor_applicable','row_id':RID,'source':{'path':str(SOURCE),'bytes':len(source),'sha256':sha(source),'global_target_start_line_zero_based':global_line},'candidate':{'packets_path':str(PACKETS),'packets_sha256':PACKETS_SHA,'selection_bytes':len(view),'selection_sha256':sha(view),'replacement_line_zero_based':local_line},'facts':{'candidate_view_reconstruction_exact':True,'candidate_hash_is_selection_hash':True,'candidate_hash_is_full_active_document_hash':False,'candidate_range_is_global_editor_range':False},'training_admission':False}
Path(__file__).with_name('actual-train-mismatch.json').write_text(json.dumps(out,indent=2,sort_keys=True)+'\n');print(json.dumps({'status':out['status'],'global':global_line,'candidate':local_line}))
