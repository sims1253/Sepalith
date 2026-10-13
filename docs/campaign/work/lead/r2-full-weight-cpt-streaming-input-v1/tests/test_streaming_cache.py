import hashlib,json,os,resource,sys,tempfile,unittest
from pathlib import Path
P=Path(__file__).resolve().parents[1];sys.path.insert(0,str(P/'source'))
from cpt_streaming_cache import StreamingCptDataset,build,sha256

def row(doc,chunk,start,payload,final,overlap=0):
 ids=[0,*payload,1];labels=[-100,*payload,1 if final else -100]
 for i in range(overlap):labels[1+i]=-100
 return {'schema':1,'row_id':f'{doc}:{chunk}','document_id':doc,'package':'pkg'+doc,'group_id':'g'+doc,'cpt_partition':'cpt_train','source_path':'TRAIN/'+doc+'.R','source_sha256':hashlib.sha256(doc.encode()).hexdigest(),'chunk_index':chunk,'input_ids':ids,'labels':labels,'attention_mask':[1]*len(ids),'source_token_start':start,'source_token_end':start+len(payload)-overlap,'overlap_context_tokens':overlap,'is_document_end':final,'supervised_tokens':sum(x!=-100 for x in labels),'document_token_count':start+len(payload)-overlap if final else 4}
class T(unittest.TestCase):
 def setUp(self):self.t=tempfile.TemporaryDirectory(dir='/mnt/e');self.root=Path(self.t.name)
 def tearDown(self):self.t.cleanup()
 def files(self,rows,draws=None):
  rp=self.root/'rows.jsonl';rp.write_text(''.join(json.dumps(x,separators=(',',':'))+'\n' for x in rows));ids=[r['row_id'] for r in rows];draws=draws or ids+ids[:(-len(ids))%16];sp=self.root/'schedule.json';sp.write_text(json.dumps({'split_id':'x','method':'one_pass_plus_named_replay_v1','effective_batch':16,'max_steps':len(draws)//16,'token_rows_sha256':sha256(rp),'row_ids':draws}));return rp,sp
 def test_exact_roundtrip_resume_positions_and_replay(self):
  rows=[row('a',0,0,[7,8],False),row('a',1,2,[8,9,10],True,1)]+[row(str(i),0,0,[20+i],True) for i in range(1,15)];rp,sp=self.files(rows);out=self.root/'cache';m=build(rp,sp,out,16,sha256(rp),sha256(sp));d=StreamingCptDataset(out,sha256(out/'manifest.json'),initial_cursor=16)
  self.assertEqual(m['counts']['rows'],16);self.assertEqual(m['counts']['documents'],15);self.assertEqual(d[0]['input_ids'],rows[0]['input_ids']);self.assertEqual(d[1]['labels'],rows[1]['labels']);self.assertEqual(d[0]['_draw_position'],0);self.assertEqual(d.initial_cursor,16);d.close()
 def test_incomplete_gap_duplicate_and_early_replay_fail(self):
  bad=[row('a',0,0,[7],False),row('a',2,1,[8],True)]
  rp,sp=self.files(bad)
  with self.assertRaisesRegex(ValueError,'chunk indexes'):build(rp,sp,self.root/'bad1',16,sha256(rp),sha256(sp))
  good=[row(str(i),0,0,[20+i],True) for i in range(16)];rp,sp=self.files(good,[good[0]['row_id']]*16)
  with self.assertRaisesRegex(ValueError,'replay before unique'):build(rp,sp,self.root/'bad2',16,sha256(rp),sha256(sp))
 def test_hash_binding_and_fresh_output(self):
  rows=[row(str(i),0,0,[20+i],True) for i in range(16)];rp,sp=self.files(rows)
  with self.assertRaisesRegex(ValueError,'source row bytes'):build(rp,sp,self.root/'bad',16,'0'*64,sha256(sp))
  out=self.root/'cache';build(rp,sp,out,16,sha256(rp),sha256(sp))
  with self.assertRaisesRegex(ValueError,'fresh'):build(rp,sp,out,16,sha256(rp),sha256(sp))
  with (out/'input_ids.i32le').open('ab') as handle:handle.write(b'X')
  with self.assertRaisesRegex(ValueError,'cache payload differs'):StreamingCptDataset(out,sha256(out/'manifest.json'))
if __name__=='__main__':unittest.main(verbosity=2)
