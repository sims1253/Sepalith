from __future__ import annotations
import json,subprocess,sys,tempfile,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'proposed-source/experiments/training'))
from campaign_reward_v2 import BufferEvidence,apply_region,score_candidate,sha_text
CTX=Path('/mnt/e/sepalith/campaign-20260915/data-work/DAT10-expanded-15008-v1/candidate-context-provenance.jsonl')
ROWS=Path('/mnt/e/sepalith/campaign-20260915/data-work/DAT10-expanded-15008-v1/candidate-token-rows.jsonl')
def parse_only(text:str)->bool:
 with tempfile.NamedTemporaryFile('w',suffix='.R') as f:
  f.write(text);f.flush();r=subprocess.run(['Rscript','--vanilla','-e','parse(file=commandArgs(trailingOnly=TRUE)[1],keep.source=FALSE)',f.name],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
 return r.returncode==0
class FramingTest(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  with CTX.open() as f:cls.context_record=json.loads(f.readline())
  with ROWS.open() as f:cls.row=json.loads(f.readline())
  assert cls.context_record['row_id']==cls.row['id']
 def test_framed_fragment_claim_is_distinct(self):
  c=self.context_record['context'];baseline=self.context_record['source_provenance']['selection_source']['document_text'];body=self.row['target_body_text']
  gold=apply_region(baseline,c['replacement_range'],body,c['document_eol'])
  self.assertFalse(parse_only(gold));self.assertTrue(parse_only(gold+'\n}'))
  evidence=BufferEvidence('framed_fragment',baseline,sha_text(baseline),parse_only(baseline),False,True,'\n}','R-4.6.1-base-parse')
  value,record=score_candidate(target_operation='replace',target_body_text=body,region_old=c['region_old'],protocol_valid=True,
   protocol_failure=None,operation='replace',body_lines=['if ('],replacement_range=c['replacement_range'],document_eol='lf',buffer=evidence,parse_probe=parse_only)
  self.assertEqual(value,-0.5);self.assertEqual(record['parse_scope'],'framed_fragment');self.assertEqual(record['candidate_parse'],'failed')
if __name__=='__main__':unittest.main()
