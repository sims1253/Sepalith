from pathlib import Path
import json,hashlib,tarfile,importlib.util,sys,datetime
base=Path('/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb'); packets=Path('/mnt/e/sepalith/campaign-20260915/data-work/DAT-04-lead-structured-initial-20260912T0442.jsonl');rows=[json.loads(x) for x in packets.read_text().splitlines()]
source=Path('/home/m0hawk/Documents/Sepalith/experiments/synthetic-data/scenarios.py');spec=importlib.util.spec_from_file_location('campaign_scenario_source_review',source);module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
def sha(b):return hashlib.sha256(b).hexdigest()
def offset(text,pos):
 lines=text.splitlines(keepends=True);line=lines[pos['line']];need=pos['character'];used=0
 for i,char in enumerate(line):
  if used==need:return sum(map(len,lines[:pos['line']]))+i
  used+=2 if ord(char)>65535 else 1
 if used==need:return sum(map(len,lines[:pos['line']]))+len(line)
 raise ValueError('invalid UTF16 offset')
proof=[]
for row in rows:
 result=row['result'];prov=result['provenance'];events=result['context']['history'];item={'id':row['row_ref']['row_id'],'family':row['family'],'history_events':len(events)}
 if events:
  path=prov['source_snapshot_path']
  if '::' in path:
   archive,member=path.split('::',1)
   with tarfile.open(archive) as f:before=f.extractfile(member).read()
  else:before=Path(path).read_bytes()
  assert sha(before)==prov['before_snapshot_sha256']
  text=before.decode('utf-8');event=events[0];assert event['range_utf16']['content_sha256']==sha(before)
  first=offset(text,event['range_utf16']['start']);last=offset(text,event['range_utf16']['end']);eol='\r\n' if '\r\n' in text else '\n'
  assert text[first:last].replace('\r\n','\n')==event['old_text']
  after=text[:first]+event['new_text'].replace('\n',eol)+text[last:];raw_after=after.encode('utf-8')
  assert sha(raw_after)==result['selection_source']['content_sha256'];assert raw_after==Path(result['selection_source']['path']).read_bytes()
  item['independent_UTF16_replay']='pass';item['after_sha256']=sha(raw_after)
  ref=row['row_ref'];raw_line=None
  with Path(ref['file']).open('rb') as f:
   for n,line in enumerate(f,1):
    if n==ref['line']:raw_line=line;break
  # DAT03 hashes the exact raw JSONL line, including its record separator.
  assert sha(raw_line)==ref['raw_line_sha256']
  raw=json.loads(raw_line);module.validate_example(raw);item['fresh_constructor_validator']='pass'
  r=result['context']['replacement_range'];a=offset(after,r['start']);b=offset(after,r['end']);assert after[a:b].replace('\r\n','\n')=='\n'.join(result['context']['region_old'])
  target='\n'.join(result['target_body']).replace('\n',eol);applied=after[:a]+target+after[b:];before_parse=module.parser.parse(after.encode());after_parse=module.parser.parse(applied.encode());item['tree_sitter_before_has_error']=before_parse.root_node.has_error;item['tree_sitter_after_has_error']=after_parse.root_node.has_error
  parent=prov['parent_identity'];description=Path('/mnt/h/sepalith/normalized')/parent['package']/parent['version']/parent['package']/'DESCRIPTION'
  if description.exists():
   db=description.read_bytes();item['license_evidence']={'path':str(description),'sha256':sha(db),'license_fields':[x for x in db.decode().splitlines() if x.startswith(('License:','License_restricts_use:','License_is_FOSS:'))]}
 item['scientific_disposition']='exclude_known_v1_description_not_supported' if row['family']=='doc_sync' else 'exclude_unobserved_architecture_change' if row['family']=='edit_pairs' else 'provisional_supported_propagation_requires_bulk_source_and_collision_checks'
 proof.append(item)
report={'task':'DAT-03/DAT-04','status':'bounded_source_review_not_registry_admission','observed_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source_constructor':str(source),'source_constructor_sha256':sha(source.read_bytes()),'packet_sha256':sha(packets.read_bytes()),'rows':proof,'reviewed_rows':len(rows),'independent_history_replays':sum(bool(r['history_events']) for r in proof),'scientific_exclusions':2,'plausible_propagation_candidates':4,'final_opened':False,'cuda_started':False}
p=base/'docs/campaign/receipts/DAT-04-lead-initial-source-review.json';p.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
