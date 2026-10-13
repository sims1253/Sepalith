import copy,hashlib,importlib.util,json,unittest
from pathlib import Path
P=Path(__file__).resolve().parents[1]
s=importlib.util.spec_from_file_location('prep64',P/'prepare64.py');M=importlib.util.module_from_spec(s);s.loader.exec_module(M)
OUT=Path('/mnt/e/sepalith/campaign-20260915/data-work/Sourcewalk-noop4100-provider-v1/fallback64-root-v1')
class Test64K(unittest.TestCase):
 def test_actual_review_partition(self):
  q,h=M.classify();self.assertEqual((len(q),len(h)),(38,67))
  self.assertEqual({x['reasons'][0] for x in q},M.QUEUE_REASONS)
  self.assertTrue(all(x['reasons'][0] not in M.QUEUE_REASONS for x in h))
 def test_materialized_manifest_and_files(self):
  m=json.loads((OUT/'manifest.json').read_text());self.assertEqual((m['rows'],m['prior_supported'],m['retained_holds']),(38,3995,67));self.assertEqual(m['closure'],'3995+38+67=4100')
  self.assertFalse(m['render_authorized']);self.assertFalse(m['training_admitted']);self.assertEqual(sum(x['rows'] for x in m['entries']),38)
  for x in m['entries']:
   p=OUT/x['path'];self.assertEqual((len(p.read_bytes()),hashlib.sha256(p.read_bytes()).hexdigest()),(x['bytes'],x['sha256']))
 def test_all_materialized_rows_are_target_free_and_unique(self):
  ids=[]
  for p in sorted((OUT/'inputs').glob('shard-*.jsonl')):
   with p.open() as stream:
    for line in stream:
     row=json.loads(line);self.assertTrue(M.validate_prediction_input(row));ids.append(row['row_id'])
  self.assertEqual(len(ids),len(set(ids)));self.assertEqual(hashlib.sha256((''.join(x+'\n' for x in sorted(ids))).encode()).hexdigest(),json.loads((OUT/'manifest.json').read_text())['queued_ids_sha256'])
 def test_partition_rejects_duplicate_and_wrong_denominator(self):
  rows=[{'row_id':'a','status':'hold','reasons':['complete_span_unresolved']},{'row_id':'b','status':'hold','reasons':['namespace_status:x']}]
  self.assertEqual(tuple(map(len,M.partition(rows,2,1))),(1,1))
  bad=copy.deepcopy(rows);bad[1]['row_id']='a'
  with self.assertRaises(AssertionError):M.partition(bad,2,1)
  with self.assertRaises(AssertionError):M.partition(rows,3,1)
 def test_target_or_gold_key_rejected(self):
  self.assertTrue(M.validate_prediction_input({'schema':'sepalith.dat10.sourcewalk-noop.prediction_input.v2','row_id':'x'}))
  for key in ('target','gold_text','labels'):
   with self.assertRaises(AssertionError):M.validate_prediction_input({'schema':'sepalith.dat10.sourcewalk-noop.prediction_input.v2','row_id':'x',key:'bad'})
 def test_runner_is_dormant_exact_policy_and_limits(self):
  text=(P/'run64_lanes.sh').read_text();self.assertIn('r2-noop4100-postrender-root-v2/run_shard.sh',text);self.assertIn('65536 2048',text);self.assertIn('3600',text)
  self.assertNotIn('node ',text);self.assertEqual(M.sha(M.PROVIDER),M.PROVIDER_SHA)
if __name__=='__main__':unittest.main()
