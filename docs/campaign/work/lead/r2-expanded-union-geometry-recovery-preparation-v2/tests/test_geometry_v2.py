import copy,importlib.util,json,unittest
from pathlib import Path
P=Path(__file__).resolve().parents[1];PLAN=P.parents[4]
def module(name,path):s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
G=module('g',P/'recover_geometry_v2.py');A=module('a',PLAN/'docs/campaign/work/lead/r2-expanded-union-audit-preparation-v2/audit_expanded_union_v1.py')
ORIGINAL=Path('/mnt/e/sepalith/campaign-20260915/data-work/RL11-15006-context-v1/context-sidecar.jsonl');SEMANTIC=Path('/mnt/e/sepalith/campaign-20260915/data-work/Semantic10948-provider-materialization-v2/selected-01/selected-contexts.jsonl');PROV=Path('/mnt/e/sepalith/campaign-20260915/data-work/Semantic10948-provider-materialization-v2/final-01/candidate-provenance.jsonl')
def first(p):
 with p.open() as stream:return json.loads(next(stream))
class T(unittest.TestCase):
 def test_exact_original_identity_and_v2_audit(self):
  x=first(ORIGINAL);out=G.recover(x,x,'context','nested_original15006',x['row_id']);g=out['source_cursor_geometry'];sp=x['source_identity']['source_provenance'];self.assertEqual(g['source_sha256'],sp['source_snapshot_sha256']);self.assertEqual(g['preedit_sha256'],sp['after_snapshot_sha256']);self.assertEqual(g['source_path'],sp['source_snapshot_path']);self.assertEqual(out['source_identity'],x['source_identity']);self.assertEqual(A.geometry_from_provenance(out),(out['source_cursor_geometry_sha256'],None))
 def test_exact_semantic_identity_and_v2_audit(self):
  x,p=first(SEMANTIC),first(PROV);self.assertEqual(x['row_id'],p['row_id']);out=G.recover(x,p,'selected_context','direct',x['row_id']);g=out['source_cursor_geometry'];self.assertEqual(g['source_sha256'],p['source_identity']['source_sha256']);self.assertEqual(g['preedit_sha256'],p['preedit_sha256']);self.assertEqual(g['source_path'],p['source_identity']['source_path']);self.assertEqual(out['source_identity'],p['source_identity']);self.assertEqual(A.geometry_from_provenance(out),(out['source_cursor_geometry_sha256'],None))
 def test_full_buffer_columns_and_lines_rejected(self):
  x=first(ORIGINAL);y=copy.deepcopy(x);y['context']['replacement_range']['end']['character']=999
  with self.assertRaisesRegex(G.RecoveryError,'column_outside_buffer'):G.recover(y,y,'context','nested_original15006',y['row_id'])
  y=copy.deepcopy(x);y['context']['replacement_range']['start']={'line':999,'character':0};y['context']['replacement_range']['end']={'line':999,'character':0}
  with self.assertRaisesRegex(G.RecoveryError,'line_outside_buffer'):G.recover(y,y,'context','nested_original15006',y['row_id'])
 def test_provenance_identity_tamper_rejected(self):
  x,p=first(SEMANTIC),copy.deepcopy(first(PROV));p['source_identity']['source_sha256']='0'*64
  out=G.recover(x,p,'selected_context','direct',x['row_id']);self.assertIsNone(A.geometry_from_provenance({**out,'source_identity':first(PROV)['source_identity']})[0])
  p=copy.deepcopy(first(PROV));p['preedit_sha256']='0'*64
  with self.assertRaisesRegex(G.RecoveryError,'preedit_range_hash_join'):G.recover(x,p,'selected_context','direct',x['row_id'])
 def test_partial_nonempty_or_shifted_boundary_rejected(self):
  x,p=first(SEMANTIC),first(PROV);y=copy.deepcopy(x);y['selected_context']['replacement_range']['end']['character']=1
  with self.assertRaisesRegex(G.RecoveryError,'partial_nonempty_range_unverified'):G.recover(y,p,'selected_context','direct',x['row_id'])
  y=copy.deepcopy(x);y['selected_context']['replacement_range']['start']['line']-=1;y['selected_context']['replacement_range']['end']['line']-=1
  with self.assertRaisesRegex(G.RecoveryError,'partial_cursor_prefix_boundary'):G.recover(y,p,'selected_context','direct',x['row_id'])
if __name__=='__main__':unittest.main()
