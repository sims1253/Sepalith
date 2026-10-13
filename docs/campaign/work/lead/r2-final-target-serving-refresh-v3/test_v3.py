#!/usr/bin/env python3
import copy,hashlib,json,sys,tempfile,unittest
from unittest import mock
import numpy as np,torch
from safetensors.torch import save_file
from pathlib import Path
P=Path(__file__).resolve().parent;sys.path.insert(0,str(P))
import analyze_refresh_v3 as ar
import paired_panel_probe_v3 as probe
import prepare_refresh_v3 as prep
import audit_gguf_norms as norm
import record_imatrix_completion as imatrix
import server_lifecycle_v3 as lifecycle
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
class V3Tests(unittest.TestCase):
 def test_panels_exact_strata_reps_and_fresh_ids(self):
  ids=[]
  for key,n,bands,reps in [('ngram',40,{'2k':20,'8k':20},1),('draft',24,{'2k':12,'8k':12},2)]:
   rows,m=probe.load_panel(P/f'{key}-panel.jsonl',P/f'{key}-panel.manifest.json');self.assertEqual(len(rows),n);self.assertEqual(m['panel']['band_counts'],bands);self.assertEqual(m['profile']['expected_requests'],n*reps);ids.append({x['id'] for x in rows})
  self.assertFalse(ids[0]&ids[1])
 def binding_result(self,profile_key,base_wall=20,cand_wall=10):
  profile=json.loads((P/'binding.template.json').read_text())['spec_profiles'][profile_key];manifest=json.loads(Path(profile['manifest_path']).read_text());binding={'target':{'identity':'T'},'tokenizer_contract':{'tokenizer_json_sha256':'a'*64},'selected_quant_artifact_key':'q4_k_m','outputs':{'q4_k_m':{'sha256':'b'*64}},'acceptance':{'point_speedup_min':1.4,'bootstrap_speedup_lower_95_min':1.0,'bootstrap_iterations':5000,'bootstrap_seed':260914},'drafts':{'released_dspark':{'sha256':'d'*64,'target_relation':'released_not_target_trained'}}};bsha='c'*64
  contract={'bos_id':0,'canonical_eos_id':1,'native_eog_ids':[1,130073],'pad_id':1,'context_size':10240,'cap':64,'greedy':True,'tokenize_flags':{'add_special':False,'parse_special':False,'with_pieces':False},'no_authored_target_tail':True}
  def result(mode,wall):
   req=[]
   for x in manifest['panel']['records']:
    for rep in range(1,profile['repetitions']+1):req.append({'row_id':x['row_id'],'phase':'measurement','rep':rep,'protocol_status':'accepted','combined_case_wall_ms':wall,'termination':{'stop':True,'stop_type':'eos','cap_hit':False},'returned_token_ids':[2,1],'raw_text':'ok','draft_n':10,'draft_n_accepted':5})
   draft=None if mode in {'ordinary','ngram_mod'} else {'key':mode,'sha256':'d'*64,'target_relation':'released_not_target_trained'}
   return {'schema_version':'sepalith.r2.serving.spec-train-panel-probe.v2','status':'completed','panel':{'manifest_sha256':sha(profile['manifest_path']),'source_sha256':manifest['panel']['sha256'],'profile':manifest['profile']},'native_contract':contract,'artifact_binding':{'binding_sha256':bsha,'target_identity':'T','tokenizer_json_sha256':'a'*64,'artifact_key':'q4_k_m','artifact_sha256':'b'*64,'mode':mode,'profile_id':profile['id'],'draft':draft},'requests':req}
  return binding,bsha,profile,result('ordinary',base_wall),result('ngram_mod' if profile_key=='ngram' else 'released_dspark',cand_wall)
 def test_complete_fast_pair_wins_and_slow_pair_honestly_has_no_winner(self):
  b,h,p,x,y=self.binding_result('ngram');r=ar.compare(b,h,p,x,y,'ngram_mod');self.assertEqual(r['promotion_status'],'candidate_meets_all_gates');self.assertGreater(r['latency_ms']['bootstrap_speedup_lower_95'],1.0);self.assertGreaterEqual(r['latency_ms']['median_speedup'],1.4);self.assertEqual(r['artifact_key'],'q4_k_m')
  b,h,p,x,y=self.binding_result('ngram',20,15);r=ar.compare(b,h,p,x,y,'ngram_mod');self.assertEqual(r['promotion_status'],'no_winner');self.assertEqual(r['status'],'complete')
  b,h,p,x,y=self.binding_result('ngram',20,10)
  for row in y['requests'][:3]:row['combined_case_wall_ms']=100
  r=ar.compare(b,h,p,x,y,'ngram_mod');self.assertFalse(r['gates']['candidate_p95_no_regression']);self.assertEqual(r['promotion_status'],'no_winner')
 def test_draft_bootstrap_clusters_trace_and_retains_both_repetitions(self):
  b,h,p,x,y=self.binding_result('draft');r=ar.compare(b,h,p,x,y,'released_dspark');self.assertEqual(r['trace_clusters'],{'2k':12,'8k':12});self.assertEqual(r['requests'],48);self.assertEqual(r['latency_ms']['bootstrap_unit'],'paired_trace_cluster_stratified_by_2k_8k')
 def test_missing_or_duplicate_request_fails(self):
  b,h,p,x,y=self.binding_result('ngram');y['requests'].pop()
  with self.assertRaisesRegex(ValueError,'coverage'):ar.compare(b,h,p,x,y,'ngram_mod')
  b,h,p,x,y=self.binding_result('ngram');y['requests'].append(copy.deepcopy(y['requests'][0]))
  with self.assertRaisesRegex(ValueError,'duplicate'):ar.compare(b,h,p,x,y,'ngram_mod')
 def test_eos_exactly_at_cap_uses_explicit_not_length_semantics(self):
  b,h,p,x,y=self.binding_result('ngram');row=y['requests'][0];row['returned_token_ids']=[7]*63+[1];row['termination']={'stop':True,'stop_type':'eos','cap_hit':False};x['requests'][0]=copy.deepcopy(row);x['requests'][0]['artifact_binding']={} if False else x['requests'][0].get('artifact_binding')
  r=ar.compare(b,h,p,x,y,'ngram_mod');self.assertEqual((r['cap_hits']['candidate']['denominator'],r['cap_hits']['candidate']['hits']),(40,0));self.assertEqual(r['cap_hits']['candidate']['by_band']['8k']['denominator'],20)
 def test_converter_closure_is_current_and_mutation_rejected(self):
  t=json.loads((P/'binding.template.json').read_text());prep.validate_converter(t['converter_source']);m=json.loads((P/'converter-source-closure.json').read_text());m['files'][0]['sha256']='0'*64
  with tempfile.TemporaryDirectory() as td:
   q=Path(td)/'m.json';q.write_text(json.dumps(m));bad=copy.deepcopy(t['converter_source']);bad['manifest']={'path':str(q),'sha256':sha(q)}
   with self.assertRaisesRegex(ValueError,'converter source'):prep.validate_converter(bad)
 def test_target_requires_dense_weight_manifest_and_exact_selection(self):
  with tempfile.TemporaryDirectory() as td:
   d=Path(td);ident={'x':1};files={}
   for name,data in [('model.safetensors',b'x'),('tokenizer.json',b't'),('config.json',b'c'),('tokenizer_config.json',b'tc'),('generation_config.json',b'gc'),('chat_template.jinja',b'ct')]:p=d/name;p.write_bytes(data);files[name]={'bytes':len(data),'sha256':sha(p)}
   manifest={'schema_version':1,'full':True,'checkpoint_kind':'full_weights','identity':ident,'files':files};mp=d/'campaign-manifest.json';mp.write_text(json.dumps(manifest));identity=hashlib.sha256(json.dumps(ident,sort_keys=True,separators=(',',':')).encode()).hexdigest();sel={'schema':'sepalith.run06.final-target-selection.v1','status':'selected_for_serving_refresh','target_identity':'T','campaign_manifest_sha256':sha(mp),'campaign_identity_sha256':identity};sp=d/'selection.json';sp.write_text(json.dumps(sel));t={'identity':'T','hf_dir':str(d),'campaign_manifest_path':str(mp),'campaign_manifest_sha256':sha(mp),'campaign_identity_sha256':identity,'weights_sha256':files['model.safetensors']['sha256'],'selection_receipt_path':str(sp),'selection_receipt_sha256':sha(sp),'required_hf_files':{k:v['sha256'] for k,v in files.items()},'expected_norm_count':85,'expected_fp32_norm_count':85,'expected_tensor_count':381};prep.validate_target(t,True);manifest['files'].pop('model.safetensors');mp.write_text(json.dumps(manifest));t['campaign_manifest_sha256']=sha(mp)
   with self.assertRaisesRegex(ValueError,'dense weights'):prep.validate_target(t,False)
 def test_norm_mapping_is_narrow(self):
  self.assertEqual(norm.mapped('model.norm.weight'),'output_norm.weight');self.assertEqual(norm.mapped('model.layers.4.input_layernorm.weight'),'blk.4.attn_norm.weight')
  with self.assertRaisesRegex(ValueError,'unreviewed'):norm.mapped('model.layers.0.some_new_norm.weight')
 def test_norm_auditor_compares_saved_f32_and_bf16_values_through_real_gguf_reader(self):
  root=Path('/home/m0hawk/Documents/Sepalith/experiments/bin/src/llamacpp-b10453');sys.path.insert(0,str(root/'gguf-py'));from gguf import GGUFWriter
  with tempfile.TemporaryDirectory() as td:
   d=Path(td);hf=d/'hf';hf.mkdir();saved={'model.norm.weight':torch.tensor([1.000000119,-.33333334],dtype=torch.float32),'model.layers.0.input_layernorm.weight':torch.tensor([1.25,-2.5],dtype=torch.bfloat16),'model.layers.0.post_attention_layernorm.weight':torch.tensor([.75,3.0],dtype=torch.float32)};save_file(saved,hf/'model.safetensors')
   gp=d/'tiny.gguf';w=GGUFWriter(str(gp),'llama');w.add_tensor('output_norm.weight',saved['model.norm.weight'].numpy());w.add_tensor('blk.0.attn_norm.weight',saved['model.layers.0.input_layernorm.weight'].float().numpy());w.add_tensor('blk.0.ffn_norm.weight',saved['model.layers.0.post_attention_layernorm.weight'].numpy());w.write_header_to_file();w.write_kv_data_to_file();w.write_tensors_to_file();w.close()
   binding={'target':{'hf_dir':str(hf),'identity':'T','expected_norm_count':3,'expected_fp32_norm_count':2,'expected_tensor_count':3,'weights_sha256':'d'*64},'outputs':{'f16':{'sha256':sha(gp)}},'converter_source':{'root':str(root)}};bp=d/'b.json';bp.write_text(json.dumps(binding))
   with mock.patch.object(norm,'validate_binding',return_value=True),mock.patch.object(norm,'artifact',return_value=gp):result=norm.audit(bp,'f16')
   self.assertTrue(result['bit_exact']);self.assertEqual((result['norms'],result['saved_fp32_norms']),(3,2));self.assertEqual({x['saved_dtype'] for x in result['rows']},{'F32','BF16'})

 def test_template_is_unselected_and_unlaunchable(self):
  b=json.loads((P/'binding.template.json').read_text());self.assertIsNone(b['target']);self.assertFalse(b['launch_authorized']);self.assertIsNone(b['root_admission'])
  with self.assertRaisesRegex(ValueError,'launch-authorized'):prep.validate_binding(b)
 def test_imatrix_receipt_requires_exit_chunks_coverage_and_exact_command(self):
  b={'calibration':{'chunks':64,'text_sha256':'c'*64},'outputs':{'imatrix':{'path':'/tmp/imatrix','sha256':'i'*64},'f16':{'sha256':'f'*64}},'tools':{'imatrix':{'sha256':'t'*64}}};execution={'schema':'sepalith.run06.root-command-terminal.v1','status':'completed','argv':['imatrix'],'command_sha256':imatrix.command_sha(['imatrix']),'exit_code':0,'processed_chunks':64,'non_embedding_coverage_mismatches':0}
  with tempfile.TemporaryDirectory() as td:
   d=Path(td);bp=d/'b.json';bp.write_text(json.dumps(b));xp=d/'x.json';xp.write_text(json.dumps(execution));out=d/'receipt.json'
   with mock.patch.object(imatrix,'validate_binding',return_value=True),mock.patch.object(imatrix,'imatrix_argv',return_value=['imatrix']),mock.patch.object(imatrix,'artifact',return_value=Path('/tmp/imatrix')),mock.patch.object(imatrix,'sha',side_effect=lambda x:'i'*64 if str(x)=='/tmp/imatrix' else sha(x)):
    imatrix.record(bp,xp,out);self.assertEqual(json.loads(out.read_text())['processed_chunks'],64)
    execution['processed_chunks']=63;xp.write_text(json.dumps(execution))
    with self.assertRaisesRegex(ValueError,'chunk count'):imatrix.record(bp,xp,d/'bad.json')
 def test_lifecycle_uses_selected_quant_and_exact_port_without_launch(self):
  b={'selected_quant_artifact_key':'iq3_m','outputs':{'iq3_m':{'path':'/tmp/model-iq3.gguf'}},'tools':{'server':{'path':'/bin/false'}},'runtime':{'common_server_argv':['--port','$PORT'],'ngram_argv':['--spec-type','ngram-mod']}}
  with mock.patch.object(lifecycle,'validate_norm_audit',return_value=True),mock.patch.object(lifecycle,'artifact',return_value=Path('/tmp/model-iq3.gguf')):
   argv=lifecycle.replace_port(lifecycle.server_argv(b,'ngram_mod'),4242)
  self.assertEqual(argv[:3],['/bin/false','-m','/tmp/model-iq3.gguf']);self.assertIn('4242',argv);self.assertNotIn('$PORT',argv)
if __name__=='__main__':unittest.main(verbosity=2)
