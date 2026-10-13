import hashlib,importlib.util,json,subprocess,tempfile,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];ANALYZER=ROOT/'source/analyze_semantics.py';RHELPER=ROOT/'source/semantic_scope.R'
spec=importlib.util.spec_from_file_location('hash_analyzer',ANALYZER);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
def sha(b):return hashlib.sha256(b).hexdigest()
def run(groups):
 with tempfile.TemporaryDirectory() as d:
  d=Path(d);payload=[];expected={}
  for i,(raw,rows) in enumerate(groups):
   p=d/f's{i}.R';p.write_bytes(raw);payload.append({'source_path':str(p),'source_sha256':sha(raw),'rows':rows});expected.update({r['row_id']:sha(raw) for r in rows})
  inp=d/'in.json';out=d/'out.jsonl';inp.write_text(json.dumps({'source_groups':payload}));x=subprocess.run(['Rscript','--vanilla',str(RHELPER),str(inp),str(out)],capture_output=True,text=True,timeout=30);assert x.returncode==0,x.stderr
  rows=[json.loads(x) for x in out.read_text().splitlines()];return rows,m.validate_scope_output(rows,expected)
class Contract(unittest.TestCase):
 def test_duplicate_target_hold_keeps_hash(self):
  raw=b'f <- function(x) x\nf <- function(y) y\n';rows,by=run([(raw,[{'row_id':'dup','target_definition_name':'f'}])]);self.assertEqual(by['dup']['status'],'hold_target_definition_not_unique');self.assertEqual(by['dup']['parsed_source_sha256'],sha(raw))
 def test_nonascii_lf_hash_exact(self):
  raw='h <- function(x) "caf\u00e9 \U0001f600"\n'.encode();_,by=run([(raw,[{'row_id':'utf8','target_definition_name':'h'}])]);self.assertEqual(by['utf8']['parsed_source_sha256'],sha(raw));self.assertEqual(by['utf8']['status'],'scope_inventory_complete')
 def test_uniform_crlf_normalized_variant_hash_exact(self):
  raw=b"#' docs\r\nf <- function(x) {\r\n x\r\n}\r\n";target=b"#' docs\n";occ,method,parsed=m.occurrence(raw,target);self.assertEqual((occ,method),(1,'uniform_crlf_to_lf'));self.assertNotEqual(sha(raw),sha(parsed));_,by=run([(parsed,[{'row_id':'crlf','target_definition_name':'f'}])]);self.assertEqual(by['crlf']['parsed_source_sha256'],sha(parsed))
 def test_multiple_variants_same_origin_stay_hash_bound(self):
  a=b'f <- function(x) x\n';b=b'f <- function(y) y\n';rows,by=run([(a,[{'row_id':'a','target_definition_name':'f'}]),(b,[{'row_id':'b','target_definition_name':'f'}])]);self.assertEqual(by['a']['parsed_source_sha256'],sha(a));self.assertEqual(by['b']['parsed_source_sha256'],sha(b));self.assertNotEqual(by['a']['parsed_source_sha256'],by['b']['parsed_source_sha256'])
 def test_parse_error_still_keeps_hash(self):
  raw=b'f <- function(\n';_,by=run([(raw,[{'row_id':'bad','target_definition_name':'f'}])]);self.assertEqual(by['bad']['status'],'hold_source_parse_error');self.assertEqual(by['bad']['parsed_source_sha256'],sha(raw))
 def test_missing_mismatch_and_id_substitution_rejected(self):
  good='a'*64
  with self.assertRaisesRegex(RuntimeError,'hash missing'):m.validate_scope_output([{'row_id':'a'}],{'a':good})
  with self.assertRaisesRegex(RuntimeError,'hash mismatch'):m.validate_scope_output([{'row_id':'a','parsed_source_sha256':'b'*64}],{'a':good})
  with self.assertRaisesRegex(RuntimeError,'ID closure'):m.validate_scope_output([{'row_id':'b','parsed_source_sha256':good}],{'a':good})
if __name__=='__main__':unittest.main()
