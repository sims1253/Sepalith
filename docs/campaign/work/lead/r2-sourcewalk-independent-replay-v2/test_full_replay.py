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
    'operation':'no_op','context':{'cursor':{'region_line_index':-1},'replacement_range':{'start':{'line':2,'character':0},'end':{'line':2,'character':0}}},'provenance':{'physical_blank_anchor':True}}}
  raw={'kind':'blank_between','region_old':[''],'region_new':[]}
  self.assertTrue(m.noop_geometry(packet,raw,text.encode())['ok'])
  packet['result']['selection_source']['content_sha256']='0'*64
  self.assertFalse(m.noop_geometry(packet,raw,text.encode())['ok'])
 def test_only_uniform_crlf_eol_normalization_is_allowed(self):
  self.assertEqual(m.window_occurrence(b'a\r\nb\r\n',b'a\nb\n'),(1,'uniform_crlf_to_lf'))
  self.assertEqual(m.window_occurrence(b'a \n',b'a\n'),(0,'no_match'))
 def test_stat_stability_is_gate(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'x';p.write_bytes(b'abc');data,e=m.stable_read(p);self.assertEqual(data,b'abc');self.assertTrue(e['stat_stable'])
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

if __name__=='__main__':unittest.main()
