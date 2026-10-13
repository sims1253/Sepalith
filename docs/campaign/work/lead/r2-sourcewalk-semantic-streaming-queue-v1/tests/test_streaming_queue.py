from __future__ import annotations
import copy,importlib.util,json,tempfile
from pathlib import Path
import unittest
HERE=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('q',HERE/'source/run_streaming_semantic_queue.py');q=importlib.util.module_from_spec(spec);spec.loader.exec_module(q)
def write(path,value):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,sort_keys=True)+'\n')
def line(path,values):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(''.join(json.dumps(x,sort_keys=True)+'\n' for x in values))
class TestQueue(unittest.TestCase):
 def fixture(self,td,duplicate=False):
  root=Path(td);replay=root/'replay';base=root/'base';q.BASE=base
  packet=base/'shard-0005/structured-materialization-v1/candidate-packets.jsonl';line(packet,[{'row_ref':{'row_id':'a'}}])
  pm=packet.parent/'manifest.json';write(pm,{'outputs':{'candidate_packets':{'path':str(packet),'sha256':q.sha(packet),'rows':1}}})
  index={'schema':'sepalith.dat10.sourcewalk_raw_index.v3','status':'complete','requested_shards':[5],'input_inventory':{'shard_pins':[{'shard':5,'sha256':'t'*64,'manifest_sha256':'m'*64}]},'index_files':[{'shard':5,'sha256':'i'*64,'rows':2,'path':'/unused'}]}
  ip=replay/'index/manifest.json';write(ip,index);index_sha=q.sha(ip)
  rows=[{'row_id':'a','shard':5,'family':'roxygen_drafting','status':q.QUEUED},{'row_id':'b','shard':5,'family':'no_op','status':'provenance_supported_candidate_root_review_required'}]
  if duplicate:rows[1]['row_id']='a'
  ledger=replay/'shards/shard-0005/ledger.jsonl';line(ledger,rows)
  binding={'driver_sha256':q.DRIVER_SHA,'index_manifest_sha256':index_sha,'index_shard_sha256':'i'*64,'token_rows_sha256':'t'*64,'token_manifest_sha256':'m'*64,'candidate_packets_sha256':q.sha(packet),'candidate_packet_manifest_sha256':q.sha(pm),'global_sha256':q.GLOBAL_SHA,'cpt_sha256':q.CPT_SHA,'hold_ledger_sha256':q.HOLD_SHA,'strict_validator_sha256':q.STRICT_SHA,'license_parser_sha256':q.LICENSE_SHA}
  receipt={'schema':'sepalith.dat10.sourcewalk_provenance_shard.v3','status':'complete','shard':5,'binding':binding,'rows':2,'outputs':[{'path':str(ledger),'rows':2,'bytes':ledger.stat().st_size,'sha256':q.sha(ledger)}]};rp=ledger.parent/'receipt.json';write(rp,receipt)
  return replay,index,{5:index['index_files'][0]},index_sha,rp,receipt
 def test_exact_committed_receipt_and_tamper_rejected(self):
  with tempfile.TemporaryDirectory() as td:
   replay,index,mapping,index_sha,rp,receipt=self.fixture(td)
   got=q.validate_receipt(replay,5,index,mapping,index_sha);self.assertEqual((got['receipt_rows'],got['queued_rows'],got['queued_ids']),(2,1,['a']))
   bad=copy.deepcopy(receipt);bad['binding']['global_sha256']='0'*64;write(rp,bad)
   with self.assertRaisesRegex(q.QueueError,'binding changed'):q.validate_receipt(replay,5,index,mapping,index_sha)
 def test_duplicate_provenance_ids_rejected(self):
  with tempfile.TemporaryDirectory() as td:
   replay,index,mapping,index_sha,*_=self.fixture(td,True)
   with self.assertRaisesRegex(q.QueueError,'duplicated'):q.validate_receipt(replay,5,index,mapping,index_sha)
 def test_partial_never_claims_global_terminal(self):
  results=[{'shard':5,'queued_ids':['a']}];sources=[{'shard':5}]
  status,closure=q.aggregate_status(results,sources,None)
  self.assertEqual(status,'partial_review_only');self.assertFalse(any(closure.values()))
 def test_global_terminal_requires_all_shards_and_exact_ids(self):
  with tempfile.TemporaryDirectory() as td:
   replay,index,mapping,index_sha,rp,receipt=self.fixture(td);terminal=replay/'manifest.json';write(terminal,{'status':'complete_review_only_no_admission','shard_receipts':[{'shard':5,'receipt_sha256':q.sha(rp)}]})
   status,closure=q.aggregate_status([{'shard':5,'queued_ids':['a']}],[{'shard':5}],terminal)
   self.assertEqual(status,'complete_review_only');self.assertTrue(all(closure.values()))
   status,_=q.aggregate_status([{'shard':5,'queued_ids':[]}],[{'shard':5}],terminal);self.assertEqual(status,'partial_review_only')
 def test_cross_shard_duplicate_semantic_id_rejected(self):
  with self.assertRaisesRegex(q.QueueError,'duplicate semantic'):
   q.aggregate_status([{'shard':1,'queued_ids':['a']},{'shard':2,'queued_ids':['a']}],[{'shard':1},{'shard':2}],None)
 def test_reuse_requires_exact_source_code_and_output(self):
  with tempfile.TemporaryDirectory() as td:
   target=Path(td)/'shard-0005';ledger=target/'semantic-ledger.jsonl';line(ledger,[{'row_id':'a'}]);source={'shard':5};code={'analyzer_sha256':'a'};write(target/'manifest.json',{'status':'complete_review_only','streaming_binding':source,'code':code,'exact_id_closure':True,'output':{'bytes':ledger.stat().st_size,'sha256':q.sha(ledger)}})
   self.assertTrue(q.reusable(target,source,code));ledger.write_text('{}\n');self.assertFalse(q.reusable(target,source,code))
 def test_shards_and_concurrency_bounds(self):
  with tempfile.TemporaryDirectory() as td:
   replay=Path(td);self.assertEqual(q.parse_shards('1,3',replay),[1,3])
   for bad in ('3,1','1,1',''):
    with self.assertRaises(q.QueueError):q.parse_shards(bad,replay)
if __name__=='__main__':unittest.main()
