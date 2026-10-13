import hashlib,json,os,shutil,subprocess,sys,tempfile,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'source'));import materialize_noop4100 as m

def row(rid,prompt,target=m.TARGET):return {'id':rid,'prompt_text':prompt,'target_text':target}
def register(candidate,geometry,state):
 pair=m.digest_text(candidate['prompt_text']+'\0'+candidate['target_text']);p=m.digest_text(candidate['prompt_text']);t=m.digest_text(candidate['target_text']);state[0][pair]=candidate['id'];state[1][p]=(t,candidate['id']);state[2][geometry]=(t,candidate['id'])
def dump(path,values):
 with Path(path).open('w')as f:
  for x in values:f.write(json.dumps(x)+'\n')

class Materializer(unittest.TestCase):
 def setUp(self):self.tmp=tempfile.TemporaryDirectory(dir='/mnt/e/sepalith/campaign-20260915/tmp');self.root=Path(self.tmp.name);self.empty=m.build_indexes([])
 def tearDown(self):self.tmp.cleanup()
 def test_same_file_different_cursor_and_all_noedit_different_files_are_retained(self):
  state=({}, {}, {});a=row('a','prompt at cursor one');b=row('b','prompt at cursor two');c=row('c','same target another file')
  self.assertEqual(m.dedup_decision(a,'1'*64,self.empty,state)[0],'candidate');register(a,'1'*64,state)
  self.assertEqual(m.dedup_decision(b,'2'*64,self.empty,state)[0],'candidate');register(b,'2'*64,state)
  self.assertEqual(m.dedup_decision(c,'3'*64,self.empty,state)[0],'candidate')
 def test_exact_prompt_target_duplicate_removed_but_target_only_not_removed(self):
  state=({}, {}, {});a=row('a','same');register(a,'1'*64,state)
  self.assertEqual(m.dedup_decision(row('b','same'),'2'*64,self.empty,state)[:2],('excluded','duplicate_candidate_prompt_target'))
  self.assertEqual(m.dedup_decision(row('c','different'),'3'*64,self.empty,state)[0],'candidate')
 def test_same_prompt_conflicting_edit_target_is_held(self):
  existing=m.build_indexes([('edit',row('edit-id','identical prompt','changed\n>>>>>>> UPDATED'))])
  self.assertEqual(m.dedup_decision(row('noop','identical prompt'),'9'*64,existing,({}, {}, {}))[:2],('hold','contradiction_existing_prompt'))
 def test_unbound_eventual9534_registry_fails_closed(self):
  registry=self.root/'registry.json';registry.write_text(json.dumps({'schema':'sepalith.dat10.noop4100.dedup-registry.v1','datasets':{'accepted_current_20191':{},'finalized_semantic10948':{},'eventual_semantic9534':None}}))
  with self.assertRaisesRegex(ValueError,'dedup manifest pin|unbound dedup'):m.registry_rows(registry,m.sha(registry))
 def real_geometry(self,rid):
  data=Path('/mnt/e/sepalith/campaign-20260915/data-work/Sourcewalk-noop4100-reconstruction-v1/inputs-full41');shard={'47a96cc61e9860688ea9114e':10,'49da0198cb6f200c0214a9ee':11}[rid]
  with (data/f'shard-{shard:04d}.jsonl').open()as stream:prediction=next(json.loads(x)for x in stream if rid in x)
  with (data/f'shard-{shard:04d}.sidecar.jsonl').open()as stream:sidecar=next(json.loads(x)for x in stream if rid in x)
  inp=self.root/f'{rid}.jsonl';out=self.root/f'{rid}.out.jsonl';dump(inp,[prediction]);provider=Path('docs/campaign/work/lead/r2-noop4100-provider-preparation-v1');base=Path('docs/campaign/work/lead/r2-semantic10948-provider-materialization-v2');cmd=[shutil.which('node'),'--no-warnings=ExperimentalWarning','--experimental-strip-types',str(provider/'render_shard.ts'),str(inp),str(out),'/home/m0hawk/Documents/Sepalith/.venv-sft/bin/python',str(base/'tokenize_bridge.py'),'/home/m0hawk/.local/state/sepalith/campaign-20260915/models/SFT-primary-step1000-theta0/tokenizer.json',str(base/'source/namespace_evidence.R'),'16384','2048'];result=subprocess.run(cmd,capture_output=True,text=True,timeout=30,env={**os.environ,'CUDA_VISIBLE_DEVICES':'','PYTHONDONTWRITEBYTECODE':'1'});self.assertEqual(result.returncode,0,result.stderr);selected=json.loads(out.read_text());return prediction,sidecar,m.protocol.PromptContext.from_mapping(selected['selected_context'])
 def test_real_nonzero_and_crlf_geometry_reapply_raw_noop_and_token_contract(self):
  tok=m.Tok('/home/m0hawk/.local/state/sepalith/campaign-20260915/models/SFT-primary-step1000-theta0/tokenizer.json')
  for rid in ('47a96cc61e9860688ea9114e','49da0198cb6f200c0214a9ee'):
   prediction,sidecar,context=self.real_geometry(rid);geometry=m.geometry_identity(context,prediction,sidecar);self.assertEqual(len(geometry),64);built=m.protocol.build_training_row(context,operation='no_op',region_new=list(context.region_old),tokenizer=tok,row_id=rid,family='no_op',package_id=sidecar['identity']['package_id'],split='train');m.validate_token_row(built,tok);self.assertEqual(built['target_text'],m.TARGET);self.assertEqual(built['input_ids'][-1],1)
 def test_tampered_cursor_and_eos_are_rejected(self):
  prediction,sidecar,context=self.real_geometry('47a96cc61e9860688ea9114e');prediction['cursor']={'line':0,'character':0}
  with self.assertRaisesRegex(ValueError,'cursor'):m.geometry_identity(context,prediction,sidecar)
  tok=m.Tok('/home/m0hawk/.local/state/sepalith/campaign-20260915/models/SFT-primary-step1000-theta0/tokenizer.json');built=m.protocol.build_training_row(context,operation='no_op',region_new=list(context.region_old),tokenizer=tok,row_id='x',family='no_op',package_id=sidecar['identity']['package_id'],split='train');built['input_ids'][-1]=2
  with self.assertRaises(ValueError):m.validate_token_row(built,tok)

if __name__=='__main__':unittest.main(verbosity=2)
