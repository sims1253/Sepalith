import hashlib,json,os,resource,sys,tempfile,unittest
from pathlib import Path
P=Path(__file__).resolve().parents[1];TRAINING=P/'source/experiments/training';sys.path.insert(0,str(TRAINING))
from cpt_streaming_cache import StreamingCptDataset,build,sha256
import bind_full_corpus_cpt as binder
import full_weight_cpt_trainer as trainer

def row(doc,chunk,start,payload,final,overlap=0):
 ids=[0,*payload,1];labels=[-100,*payload,1 if final else -100]
 for i in range(overlap):labels[1+i]=-100
 return {'schema':1,'row_id':f'{doc}:{chunk}','document_id':doc,'package':'pkg'+doc,'group_id':'g'+doc,'cpt_partition':'cpt_train','source_path':'TRAIN/'+doc+'.R','source_sha256':hashlib.sha256(doc.encode()).hexdigest(),'chunk_index':chunk,'input_ids':ids,'labels':labels,'attention_mask':[1]*len(ids),'source_token_start':start,'source_token_end':start+len(payload)-overlap,'overlap_context_tokens':overlap,'is_document_end':final,'supervised_tokens':sum(x!=-100 for x in labels),'document_token_count':start+len(payload)-overlap if final else 4}
class T(unittest.TestCase):
 def setUp(self):self.t=tempfile.TemporaryDirectory(dir='/mnt/e');self.root=Path(self.t.name)
 def tearDown(self):self.t.cleanup()
 def files(self,rows,draws=None):
  rp=self.root/'rows.jsonl';rp.write_text(''.join(json.dumps(x,separators=(',',':'))+'\n' for x in rows));ids=[r['row_id'] for r in rows];draws=draws or ids+ids[:(-len(ids))%16];replay=len(draws)-len(ids);sp=self.root/'schedule.json';sp.write_text(json.dumps({'split_id':'x','method':'one_pass_plus_named_replay_v1','effective_batch':16,'max_steps':len(draws)//16,'token_rows_sha256':sha256(rp),'row_ids':draws,'replay_count':replay,'replay_row_ids':draws[-replay:] if replay else []}));return rp,sp
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
  with self.assertRaisesRegex(ValueError,'source row streamed bytes'):build(rp,sp,self.root/'bad',16,'0'*64,sha256(sp))
  out=self.root/'cache';build(rp,sp,out,16,sha256(rp),sha256(sp))
  with self.assertRaisesRegex(ValueError,'fresh'):build(rp,sp,out,16,sha256(rp),sha256(sp))
  with (out/'input_ids.i32le').open('ab') as handle:handle.write(b'X')
  with self.assertRaisesRegex(ValueError,'cache payload differs'):StreamingCptDataset(out,sha256(out/'manifest.json'))
 def test_bad_carry_declared_hash_and_replay_tail_fail(self):
  rows=[row('a',0,0,[7,8],False),row('a',1,2,[99,9],True,1)]+[row(str(i),0,0,[20+i],True) for i in range(1,15)];rp,sp=self.files(rows)
  with self.assertRaisesRegex(ValueError,'carry token differs'):build(rp,sp,self.root/'carry',16,sha256(rp),sha256(sp))
  rows=[row(str(i),0,0,[20+i],True) for i in range(16)];rows[0]['token_stream_sha256']='0'*64;rp,sp=self.files(rows)
  with self.assertRaisesRegex(ValueError,'declared token-stream hash differs'):build(rp,sp,self.root/'streamhash',16,sha256(rp),sha256(sp))
  rows=[row(str(i),0,0,[20+i],True) for i in range(15)];rp,sp=self.files(rows);value=json.loads(sp.read_text());value['replay_row_ids']=['wrong'];sp.write_text(json.dumps(value))
  with self.assertRaisesRegex(ValueError,'named replay tail differs'):build(rp,sp,self.root/'replay',16,sha256(rp),sha256(sp))
 def test_representative_cache_parity_without_full_token_materialization(self):
  cache=Path('/mnt/e/sepalith/campaign-20260915/data-work/CPT-streaming-input-v1/representative-cache');self.assertEqual(sha256(cache/'manifest.json'),'96b4f7b1cdb975e4aa8a1c18ff5fd8fab1661c1169aa36c0cb6530242ea7f530');manifest=json.loads((cache/'manifest.json').read_text());schedule=json.loads(Path(manifest['source']['draw_schedule']['path']).read_text())['row_ids'];positions=[0,1,100,len(schedule)-1];wanted={schedule[i] for i in positions};actual={}
  with Path(manifest['source']['rows']['path']).open() as f:
   for line in f:
    value=json.loads(line)
    if value['row_id'] in wanted:actual[value['row_id']]=value
  d=StreamingCptDataset(cache,sha256(cache/'manifest.json'),initial_cursor=384)
  for position in positions:
   self.assertEqual(d[position]['input_ids'],actual[schedule[position]]['input_ids']);self.assertEqual(d[position]['labels'],actual[schedule[position]]['labels']);self.assertEqual(d[position]['_draw_position'],position)
  self.assertEqual(d.initial_cursor,384);d.close()
 def test_full_trainer_uses_streaming_and_generic_resume_boundary(self):
  source=(TRAINING/'full_weight_cpt_trainer.py').read_text();self.assertNotIn('class FrozenTokenRowDataset',source);self.assertIn('dataset_from_bound_recipe',source);self.assertLess(source.index('dataset, run_preflight = _validated_cohort'),source.index('from unsloth import FastLanguageModel'))
  self.assertEqual(trainer.milestone_action(128,0,{'mandatory_stop_step':128,'max_steps':4096}),{'save':True,'stop':True,'cursor':2048});self.assertEqual(trainer.milestone_action(128,2048,{'mandatory_stop_step':128,'max_steps':4096}),{'save':True,'stop':False,'cursor':2048})
 def test_missing_admission_and_fixed_horizon_are_rejected(self):
  template=self.root/'template.json';template.write_text(json.dumps({'schema':'sepalith.sft11.full-weight-cpt-full-corpus-template.v1','launch_authorized':False}));admission=self.root/'admission.json';admission.write_text(json.dumps({'schema':'sepalith.sft11.full-weight-cpt-full-corpus-root-admission.v1','status':'pending','launch_authorized':False}))
  with self.assertRaisesRegex(ValueError,'root admission missing'):binder.bind(template,admission,self.root/'bound.json')
  source=(TRAINING/'bind_full_corpus_cpt.py').read_text();self.assertNotIn('==66',source);self.assertNotIn('1024',source)
if __name__=='__main__':unittest.main(verbosity=2)
