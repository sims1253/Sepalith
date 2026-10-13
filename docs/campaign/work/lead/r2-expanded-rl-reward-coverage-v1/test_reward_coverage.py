from __future__ import annotations
import hashlib,json,sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from reward_coverage import adapt_evidence,score_gate
SIDE=Path('/mnt/e/sepalith/campaign-20260915/data-work/RL11-expanded-buffer-v5/reward-buffer-sidecar.jsonl')
PIN='126a9654b9d7d788c6cd60a4c90c9e0bf7f88fa718232be3f209473c94a81c4f'
class CoverageTest(unittest.TestCase):
 def test_all_15008_retained_with_explicit_modes(self):
  self.assertEqual(hashlib.sha256(SIDE.read_bytes()).hexdigest(),PIN);counts={};ids=set()
  with SIDE.open() as f:
   for line in f:
    x=adapt_evidence(json.loads(line));ids.add(x['row_id']);m=x['syntax_evidence_mode'];counts[m]=counts.get(m,0)+1
  self.assertEqual(len(ids),15008);self.assertEqual(counts,{'complete_document':4488,'completion_prefix':4282,'framed_fragment':3503,'unverified':2735})
 def test_unavailable_syntax_never_adds_reward(self):
  self.assertEqual(score_gate(exact=False,protocol_valid=True,false_noop_edit=False,repetition=False,syntax_status='unavailable'),(0.0,'syntax_unavailable_no_credit'))
 def test_exact_protocol_and_noop_restraint_do_not_depend_on_syntax_mode(self):
  for status in ('passed','failed','unavailable'):
   self.assertEqual(score_gate(exact=True,protocol_valid=True,false_noop_edit=False,repetition=False,syntax_status=status)[0],1.2)
   self.assertEqual(score_gate(exact=False,protocol_valid=False,false_noop_edit=False,repetition=False,syntax_status=status)[0],-1.0)
   self.assertEqual(score_gate(exact=False,protocol_valid=True,false_noop_edit=True,repetition=False,syntax_status=status)[0],-1.0)
 def test_only_verified_failure_gets_syntax_penalty(self):
  self.assertEqual(score_gate(exact=False,protocol_valid=True,false_noop_edit=False,repetition=False,syntax_status='failed')[0],-0.5)
  self.assertEqual(score_gate(exact=False,protocol_valid=True,false_noop_edit=False,repetition=False,syntax_status='passed')[0],0.0)
if __name__=='__main__':unittest.main()
