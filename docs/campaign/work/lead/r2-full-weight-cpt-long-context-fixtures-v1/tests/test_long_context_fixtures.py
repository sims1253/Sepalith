import hashlib, importlib.util, json, shutil, sys, tempfile, unittest
from pathlib import Path

PACKET=Path(__file__).resolve().parents[1]; SOURCE=PACKET/'source'; MATERIALIZED=PACKET/'materialized'
sys.path.insert(0,str(SOURCE))
import prepare_fixtures as prep
import lossless_rechunk as rechunk
import campaign_cpt_data as cpt

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rows(path): return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]

def smoke_module():
 spec=importlib.util.spec_from_file_location('test_smoke',prep.SMOKE); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m

class LongContextFixtureTests(unittest.TestCase):
 def test_two_exact_full_continuation_rows_per_context(self):
  manifest=json.loads((MATERIALIZED/'fixture-manifest.json').read_text())
  for size in (8192,16384):
   path=MATERIALIZED/'fixtures'/f'cpt-smoke-ctx{size}-2rows.jsonl'; values=rows(path)
   self.assertEqual(len(values),2); self.assertEqual(sha(path),manifest['fixtures'][str(size)]['sha256'])
   for row in values:
    self.assertEqual(len(row['input_ids']),size); self.assertEqual(row['attention_mask'],[1]*size)
    self.assertEqual(row['chunk_index'],1); self.assertEqual(row['overlap_context_tokens'],1)
    self.assertFalse(row['is_document_end']); self.assertEqual(row['labels'][:2],[-100,-100]); self.assertEqual(row['labels'][-1],-100)
    self.assertEqual(row['labels'][2:-1],row['input_ids'][2:-1]); self.assertEqual(row['supervised_tokens'],size-3)
    cpt.validate_materialized_row(row,max_sequence_tokens=size)

 def test_complete_source_reassembly_and_rechunk_conserve_every_token(self):
  result=json.loads((MATERIALIZED/'rechunked/result.json').read_text())
  manifest=json.loads((MATERIALIZED/'fixture-manifest.json').read_text())
  total=0
  for item in manifest['source_documents']:
   original=rows(item['selected_original']['path']); common,payload,ids=rechunk.validate_and_reassemble(original)
   self.assertEqual(len(payload),item['document_tokens']); self.assertEqual(common['document_id'],item['document_id']); total+=len(payload)
  self.assertEqual(total,70217)
  for size in ('8192','16384','32768'):
   self.assertEqual(result['outputs'][size]['payload_tokens'],total); self.assertEqual(result['outputs'][size]['terminal_eos'],2)
  self.assertTrue(result['guarantees']['all_source_tokens_once_per_context'])

 def test_deterministic_rebuild_reproduces_token_artifacts(self):
  expected=json.loads((MATERIALIZED/'fixture-manifest.json').read_text())
  expected_rechunk=json.loads((MATERIALIZED/'rechunked/result.json').read_text())
  with tempfile.TemporaryDirectory() as value:
   output=Path(value)/'rebuilt'; observed=prep.prepare(output)
   for size in ('8192','16384'):
    self.assertEqual(observed['fixtures'][size]['sha256'],expected['fixtures'][size]['sha256'])
   rebuilt=json.loads((output/'rechunked/result.json').read_text())
   for name in ('cpt_train_ctx8192.jsonl','cpt_train_ctx16384.jsonl','cpt_train_ctx32768.jsonl'):
    self.assertEqual(rebuilt['artifacts'][name]['sha256'],expected_rechunk['artifacts'][name]['sha256'])

 def test_actual_smoke_reader_accepts_and_cap_rejects_without_truncation(self):
  smoke=smoke_module()
  for size in (8192,16384):
   path=MATERIALIZED/'fixtures'/f'cpt-smoke-ctx{size}-2rows.jsonl'
   self.assertEqual(len(smoke.read_rows(path,size)),2)
   with self.assertRaisesRegex(ValueError,'must never truncate'): smoke.read_rows(path,size-1)

 def test_label_or_identity_tampering_is_rejected(self):
  row=rows(MATERIALIZED/'fixtures/cpt-smoke-ctx8192-2rows.jsonl')[0]
  changed=json.loads(json.dumps(row)); changed['labels'][2]=-100
  with self.assertRaises(Exception): prep.prove_fixture(changed,8192)
  changed=json.loads(json.dumps(row)); changed['source_sha256']='0'*64
  with self.assertRaises(Exception): prep.prove_fixture(changed,8192)

if __name__=='__main__': unittest.main(verbosity=2)
