"""Meaningful CPU tests for schedule coverage, geometry, and replay ordering."""
import hashlib,json,tempfile,unittest
from pathlib import Path
from full_coverage_schedule import build,load_rows
H=Path(__file__).resolve().parent
MANIFEST=H/'full-coverage-15008-draw-manifest.json';AUDIT=H/'e-actual-schedule-audit.json'

def token_row(rid,noop=False,target_body=8,total_prompt=12):
 body=list(range(100,100+target_body));terminal=[901,902,903,904,905];prompt=list(range(10,10+total_prompt));tokens=[0,*prompt,*body,*terminal,1]
 return {'id':rid,'split':'train','bos_token_id':0,'eos_token_id':1,'input_ids':tokens,'prompt_token_count':len(prompt),'target_start':len(prompt)+1,'target_body_token_count':len(body),'target_body_tokens':body,'target_terminal_token_count':len(terminal),'target_terminal_tokens':terminal,'target_token_count':len(body)+len(terminal),'family':'no_op' if noop else 'finish_block','package_id':'p','target_operation':'no_op' if noop else 'replace'}
def load_synthetic(rows,max_sequence=4096):
 with tempfile.TemporaryDirectory() as d:
  p=Path(d)/'rows.jsonl';p.write_text(''.join(json.dumps(r)+'\n' for r in rows));digest=hashlib.sha256(p.read_bytes()).hexdigest();return load_rows(p,digest,[512,1024,2048,max_sequence])

class Tests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):cls.m=json.loads(MANIFEST.read_text());cls.a=json.loads(AUDIT.read_text())
 def test_concrete_15008_exact_denominators(self):
  m=self.m;self.assertEqual((m['input']['sha256'],m['coverage']['eligible_rows'],m['coverage']['eligible_edit_rows'],m['coverage']['eligible_noop_rows']),('f4c93a760b328f154087bc036e5ef4f36f4dc37e204499abfc8fa270c1fc67cc',15008,13914,1094));self.assertEqual((m['max_steps'],m['draw_count']), (1160,18560));self.assertEqual(len(set(m['row_ids'])),15008);self.assertEqual(m['excluded'],[])
 def test_every_batch_has_explicit_25pct_noop_and_actual_bucket(self):
  draws=self.m['draws'];self.assertEqual(len(draws)%16,0)
  for start in range(0,len(draws),16):
   batch=draws[start:start+16];self.assertEqual(sum(d['semantic_noop'] for d in batch),4);self.assertEqual(len({d['batch_sequence_bucket'] for d in batch}),1);self.assertGreaterEqual(batch[0]['batch_sequence_bucket'],max(d['total_tokens'] for d in batch))
 def test_full_targets_are_scheduled_without_truncation(self):
  for d in self.m['draws']:
   self.assertEqual(d['total_tokens'],1+d['prompt_tokens']+d['supervised_target_tokens']);self.assertEqual(d['supervised_target_tokens'],d['target_body_tokens']+d['target_protocol_tokens']+1)
  self.assertEqual(self.m['token_denominators']['unique']['supervised_target_tokens_including_eos'],1576417)
 def test_unique_pool_members_precede_named_replays(self):
  c=self.m['coverage'];self.assertGreater(c['first_replay_draw_by_pool']['noop'],c['last_unique_draw_by_pool']['noop']);self.assertGreater(c['first_replay_draw_by_pool']['edit'],c['last_unique_draw_by_pool']['edit']);self.assertEqual(c['replay_draws_by_reason'],{'noop_ratio_length_alignment_replay':3546,'edit_batch_completion_alignment_replay':6})
 def test_actual_e_schedule_audit(self):
  a=self.a;self.assertEqual(a['status'],'PASS');self.assertTrue(all(a['checks'].values()));self.assertEqual(a['coverage']['milestone_distinct_rows'],{'250':4000,'500':7094,'750':10094,'868':11505,'1000':11505});self.assertEqual(a['geometry']['targets_over_192'],1963);self.assertEqual(a['geometry']['max_supervised_target_tokens_including_eos'],933)
 def test_target_over_1024_is_retained_when_sequence_fits(self):
  rows,excluded=load_synthetic([token_row('edit-long',target_body=1100),token_row('noop-long',noop=True,target_body=1100)]);self.assertEqual(excluded,[]);result=build(rows,excluded,7,2,1,[512,1024,2048,4096]);self.assertEqual(result['checks']['targets_over_1024_retained'],2);self.assertEqual({d['supervised_target_tokens'] for d in result['draws']},{1106})
 def test_sequence_over_context_is_explicitly_excluded(self):
  rows,excluded=load_synthetic([token_row('too-long',target_body=4100)],4096);self.assertEqual(rows,[]);self.assertEqual(excluded[0]['reason'],'sequence_exceeds_max')
 def test_horizon_is_pool_derived_not_fixed_1000(self):
  source=[token_row('e'+str(i)) for i in range(13)]+[token_row('n'+str(i),noop=True) for i in range(2)];rows,excluded=load_synthetic(source);result=build(rows,excluded,11,4,1,[512,1024,2048,4096]);self.assertEqual(result['max_steps'],5);self.assertEqual(result['coverage']['distinct_rows_drawn'],15)
 def test_deterministic_for_same_pool_seed_and_policy(self):
  source=[token_row('e'+str(i)) for i in range(5)]+[token_row('n'+str(i),noop=True) for i in range(2)];rows,excluded=load_synthetic(source);a=build(rows,excluded,19,4,1,[512,1024,2048,4096]);b=build(rows,excluded,19,4,1,[512,1024,2048,4096]);self.assertEqual(a,b)
 def test_malformed_terminal_eos_rejected(self):
  row=token_row('bad');row['input_ids'][-1]=99
  with self.assertRaisesRegex(ValueError,'malformed'):load_synthetic([row])
if __name__=='__main__':unittest.main(verbosity=2)
