"""Synthetic CPU controls for source slicing, canonical detection and token caps."""
import os,sys,unittest
from pathlib import Path
os.sched_setaffinity(0,set(sorted(os.sched_getaffinity(0))[:2]));sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parent))
import build_short_candidates as b

def meta(src):
 return {'path':'/synthetic-only/fixture.R','sha256':b.sha(src),'sha1':'a'*40,'git_blob_sha1':'b'*40,'document_id':b.sha(src),'group_id':'synthetic-test-group','package':'synthetic-test','version':'1','cpt_partition':'cpt_train','split':'train_group'}

class ShortTests(unittest.TestCase):
 def test_literal_tail_includes_real_closing_brace(self):
  src=b'f <- function(x) {\n  y <- x + 1\n  return(y)\n}\n';rows=list(b.tail_candidates(meta(src),src));self.assertEqual(len(rows),1);p=rows[0];pr=p['source_provenance'];self.assertEqual(pr['selection_source']['document_text']+p['row']['target_body_text'],pr['gold_applied_document_text']);self.assertTrue(p['row']['target_body_text'].endswith('}'));self.assertTrue(pr['after_R_parse']);self.assertFalse(pr['before_R_parse'])
 def test_exact_source_slice_offsets(self):
  src=b'# source heading\nf <- function(x) {\n  y <- x + 1\n  y\n}\n';p=list(b.tail_candidates(meta(src),src))[0];pr=p['source_provenance'];self.assertEqual(src[pr['target_source_byte_start']:pr['target_source_byte_end']].decode(),p['row']['target_body_text'])
 def test_no_unknown_tail_identifier(self):
  src=b'f <- function(x) {\n  y <- x + 1\n  return(unknown)\n}\n';self.assertEqual(list(b.tail_candidates(meta(src),src)),[])
 def test_no_signature_only_body(self):
  src=b'f <- function(x) {\n  return(x)\n}\n';self.assertEqual(list(b.tail_candidates(meta(src),src)),[])
 def test_no_same_line_cut(self):
  src=b'f <- function(x) { y <- x; return(y) }\n';self.assertEqual(list(b.tail_candidates(meta(src),src)),[])
 def test_real_tokenizer_protocol_and_inclusive_budget(self):
  src=b'f <- function(x) {\n  y <- x + 1\n  y\n}\n';p=list(b.tail_candidates(meta(src),src))[0];r=p['row'];self.assertEqual(r['input_ids'][0],0);self.assertEqual(r['input_ids'][-1],1);self.assertLessEqual(r['target_token_count']+1,192);self.assertEqual(b._backend.decode(r['input_ids'][1:-1],skip_special_tokens=False),r['prompt_text']+r['target_text'])
 def test_overlong_tail_rejected_not_truncated(self):
  src=('f <- function(x) {\n  y <- c(x)\n  return(c('+','.join(['y']*240)+'))\n}\n').encode()
  with self.assertRaisesRegex(ValueError,'target_over_192'):list(b.tail_candidates(meta(src),src))
 def test_long_prefix_not_silently_dropped(self):
  src=('f <- function(x) {\n  y <- x\n'+'  # '+('comment '*1300)+'\n  y\n}\n').encode()
  with self.assertRaisesRegex(ValueError,'selection_drops'):list(b.tail_candidates(meta(src),src))
 def test_pair_replays_both_edits_and_stops(self):
  src=b'f <- function(x) {\n  a <- x %>% mean()\n  a %>% sqrt()\n}\n';pair=b.pipe_pair(meta(src),src);self.assertEqual(len(pair),2);self.assertEqual(pair[0]['row']['family'],'pipe_rewrite');self.assertEqual(pair[1]['row']['target_operation'],'no_op');self.assertEqual(pair[1]['source_provenance']['remaining_eligible_sites'],0);self.assertEqual(len(pair[1]['context']['history']),2);self.assertEqual(pair[0]['source_provenance']['gold_applied_document_text'],pair[1]['source_provenance']['selection_source']['document_text'])
 def test_one_remaining_site_is_not_done(self):
  src=b'f <- function(x) {\n x %>% mean()\n}\n';bundle=b.scenario.Bundle('synthetic','R/fixture.R',src);self.assertEqual(b.scenario.extract_pipe(bundle,b.random.Random(0)),[]);self.assertEqual(len(b.eligible_pipe_sites(bundle)),1);self.assertEqual(b.pipe_pair(meta(src),src),[])
 def test_three_sites_no_false_completed_pair(self):
  src=b'f <- function(x) {\n a <- x %>% mean()\n b <- a %>% sqrt()\n b %>% abs()\n}\n';self.assertEqual(b.pipe_pair(meta(src),src),[])
 def test_placeholder_not_rewritten(self):
  src=b'f <- function(x) {\n x %>% mean(., na.rm=TRUE)\n}\n';self.assertEqual(b.eligible_pipe_sites(b.scenario.Bundle('synthetic','R/f.R',src)),[])
 def test_string_pipe_is_not_site(self):
  src=b'f <- function() { "%>%" }';self.assertEqual(b.eligible_pipe_sites(b.scenario.Bundle('synthetic','R/f.R',src)),[])
 def test_candidate_not_training_admitted(self):
  src=b'f <- function(x) {\n y <- x\n y\n}\n';p=list(b.tail_candidates(meta(src),src))[0];self.assertFalse(p['admitted_for_training']);self.assertNotIn('note',p['context']);self.assertEqual(p['context']['history'],[])
 def test_no_model_framework(self):
  self.assertNotIn('torch',sys.modules);self.assertNotIn('transformers',sys.modules)

if __name__=='__main__':unittest.main(verbosity=2)
