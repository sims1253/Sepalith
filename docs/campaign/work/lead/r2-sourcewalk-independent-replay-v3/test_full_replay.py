import importlib.util, json, tempfile, unittest
from pathlib import Path

HERE=Path(__file__).parent
spec=importlib.util.spec_from_file_location('full_replay',HERE/'full_replay.py'); m=importlib.util.module_from_spec(spec); assert spec.loader; spec.loader.exec_module(m)

class TestReplay(unittest.TestCase):
 def test_positive_license_requires_fields_and_reviewed_family(self):
  class P:
   @staticmethod
   def allowed_license(x): return x in {'MIT','GPL-3'}
  self.assertTrue(m.positive_license(b'Package: p\nLicense: MIT\n','p',P)['ok'])
  for raw in (b'Package: p\n',b'License: MIT\n',b'Package: p\nLicense: Mystery\n',b'Package: p\nLicense: MIT\nLicense_restricts_use: yes\n'):
   self.assertFalse(m.positive_license(raw,'p',P)['ok'])
 def test_noop_requires_source_backed_geometry(self):
  text='x <- 1\n\n'; packet={'result':{'selection_source':{'text':text,'content_sha256':m.sha_bytes(text.encode())},'target_body':[],
    'operation':'no_op','context':{'prefix':['x <- 1',''],'region_old':[],'suffix_lines':[],'cursor':{'region_line_index':-1},'replacement_range':{'start':{'line':2,'character':0},'end':{'line':2,'character':0}}},'provenance':{'physical_blank_anchor':True}}}
  raw={'kind':'blank_between','region_old':[''],'region_new':[]}
  self.assertTrue(m.noop_geometry(packet,raw,text.encode())['ok'])
  packet['result']['selection_source']['content_sha256']='0'*64
  self.assertFalse(m.noop_geometry(packet,raw,text.encode())['ok'])
 def test_noop_rejects_missing_or_untyped_positions(self):
  text='x <- 1\n\n';base={'result':{'selection_source':{'text':text,'content_sha256':m.sha_bytes(text.encode())},'target_body':[],'operation':'no_op','context':{'prefix':['x <- 1',''],'region_old':[],'suffix_lines':[],'cursor':{'region_line_index':-1},'replacement_range':{'start':{'line':2,'character':0},'end':{'line':2,'character':0}}},'provenance':{}}}
  raw={'kind':'blank_between','region_old':[''],'region_new':[]}
  base['result']['context']['replacement_range']['start']=None
  self.assertFalse(m.noop_geometry(base,raw,text.encode())['ok'])
 def test_global_group_requires_package_file_source_family_and_relative_path(self):
  group={'identity_forms':['pkg:p'],'files':['raw.jsonl'],'source_counts':{'scenario_no_op':1},'families':{'no_op':1}}
  ref={'file':'raw.jsonl','source':'scenario_no_op'}
  self.assertTrue(m.group_source_membership(group,'p',ref,'no_op','R/x.R','R/x.R'))
  for bad in ({**group,'files':[]},{**group,'identity_forms':[]},{**group,'source_counts':{}},{**group,'families':{}}):
   self.assertFalse(m.group_source_membership(bad,'p',ref,'no_op','R/x.R','R/x.R'))
  self.assertFalse(m.group_source_membership(group,'p',ref,'no_op','R/y.R','R/x.R'))
 def test_partition_must_be_exact_cpt_train(self):
  self.assertTrue(m.eligible_partition('cpt_train'))
  for value in (None,'','cpt_validation','unknown','train_group'):self.assertFalse(m.eligible_partition(value))
 def test_only_uniform_crlf_eol_normalization_is_allowed(self):
  self.assertEqual(m.window_occurrence(b'a\r\nb\r\n',b'a\nb\n'),(1,'uniform_crlf_to_lf'))
  self.assertEqual(m.window_occurrence(b'a \n',b'a\n'),(0,'no_match'))
 def test_stat_stability_is_gate(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'x';p.write_bytes(b'abc');data,e=m.stable_read(p);self.assertEqual(data,b'abc');self.assertTrue(e['stat_stable'])
 def test_existing_index_reuse_and_raw_stat_change_rejection(self):
  with tempfile.TemporaryDirectory() as d:
   out=Path(d);(out/'index').mkdir();raw=out/'raw.jsonl';raw.write_text('{}\n');state=raw.stat();idx=out/'index/shard-0000.jsonl';idx.write_text('')
   entry={'shard':0,'path':str(idx),'rows':0,'bytes':0,'sha256':m.sha(idx)}
   manifest={'status':'complete','driver_sha256':m.sha(Path(m.__file__).resolve()),'requested_shards':[0],'index_files':[entry],
    'raw_sources':[{'path':str(raw),'stat':{'device':state.st_dev,'inode':state.st_ino,'size':state.st_size,'mtime_ns':state.st_mtime_ns}}],
    'input_inventory':{'hold_ledger':{'sha256':m.HOLD_SHA}}}
   m.atomic_json(out/'index/manifest.json',manifest);self.assertEqual(m.validate_existing_index(out,[0])['status'],'complete')
   raw.write_text('{"changed":true}\n')
   with self.assertRaisesRegex(RuntimeError,'raw_source_stat_changed'):m.validate_existing_index(out,[0])
 def test_two_shard_interrupt_resume_exactly_once(self):
  with tempfile.TemporaryDirectory() as d:
   out=Path(d); bind={'driver':'x','input':'y'}
   self.assertEqual(m.commit_shard(out,0,bind,[{'id':'a'},{'id':'b'}]),'committed')
   self.assertEqual(m.commit_shard(out,0,bind,[{'id':'SHOULD_NOT_WRITE'}]),'reused')
   with self.assertRaises(KeyboardInterrupt):m.commit_shard(out,1,bind,[{'id':'c'}],stop_after_write=True)
   self.assertFalse((out/'shards/shard-0001').exists())
   self.assertEqual(m.commit_shard(out,1,bind,[{'id':'c'}]),'committed')
   ids=[]
   for s in (0,1): ids += [json.loads(x)['id'] for x in (out/'shards'/f'shard-{s:04d}'/'ledger.jsonl').read_text().splitlines()]
   self.assertEqual(ids,['a','b','c']);self.assertEqual(len(ids),len(set(ids)))
 def test_receipt_tamper_prevents_resume(self):
  with tempfile.TemporaryDirectory() as d:
   out=Path(d); bind={'x':1};m.commit_shard(out,0,bind,[{'id':'a'}]);(out/'shards/shard-0000/ledger.jsonl').write_text('{}\n')
   self.assertFalse(m.receipt_reusable(out/'shards/shard-0000/receipt.json',bind))
 def test_merge_rejects_same_count_substituted_id(self):
  with tempfile.TemporaryDirectory() as d:
   out=Path(d);(out/'index').mkdir(parents=True);base=out/'base';packet_dir=base/'shard-0000/structured-materialization-v1';packet_dir.mkdir(parents=True)
   packet_manifest={'outputs':{'candidate_packets':{'sha256':'p'*64}}};m.atomic_json(packet_dir/'manifest.json',packet_manifest)
   index_rows=out/'index/shard-0000.jsonl';index_rows.write_text(json.dumps({'selected':{'row_id':'expected'}})+'\n')
   entry={'shard':0,'path':str(index_rows),'rows':1,'bytes':index_rows.stat().st_size,'sha256':m.sha(index_rows)}
   pin={'shard':0,'sha256':'t'*64,'manifest_sha256':'u'*64};manifest={'status':'complete','requested_shards':[0],'index_files':[entry],'input_inventory':{'shard_pins':[pin]}};m.atomic_json(out/'index/manifest.json',manifest)
   binding={'driver_sha256':m.sha(Path(m.__file__).resolve()),'index_manifest_sha256':m.sha(out/'index/manifest.json'),'index_shard_sha256':entry['sha256'],'token_rows_sha256':pin['sha256'],'token_manifest_sha256':pin['manifest_sha256'],'candidate_packets_sha256':'p'*64,'candidate_packet_manifest_sha256':m.sha(packet_dir/'manifest.json'),'global_sha256':m.GLOBAL_SHA,'cpt_sha256':m.CPT_SHA,'hold_ledger_sha256':m.HOLD_SHA,'strict_validator_sha256':m.STRICT_SHA,'license_parser_sha256':m.LICENSE_SHA}
   m.commit_shard(out,0,binding,[{'row_id':'substitute','family':'no_op','status':'x','upstream_mechanical_hold_reasons':[]}])
   old=m.BASE;m.BASE=base
   try:
    with self.assertRaisesRegex(RuntimeError,'exact_id_closure'):m.merge(out)
   finally:m.BASE=old

if __name__=='__main__':unittest.main()
