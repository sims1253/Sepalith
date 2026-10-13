#!/usr/bin/env python3
import argparse,importlib,json,sys,tempfile,unittest
from pathlib import Path
AP=argparse.ArgumentParser(add_help=False);AP.add_argument('--arm',type=Path,required=True);ARGS,REST=AP.parse_known_args();sys.argv=[sys.argv[0],*REST]
sys.path.insert(0,str(ARGS.arm/'native_evaluator'))
import cap_contract,native_identity,native_transport as nt
class Parsed:
 status='accepted';operation='edit';body=['x <- 1'];reason=None
class Protocol:
 def valid_generation_tokens(self,t): return bool(t) and t[-1]==1 and 1 not in t[:-1] and 130073 not in t[:-1]
 def parse_output(self,text,ctx): return Parsed()
class Tok:
 def decode(self,ids,**kwargs): return 'ok'
CASE={'tokenizer':Tok(),'context':object(),'expected_region':['x <- 1'],'expected_noop':False}
def stream(tokens,stop='eos',truncated=False,text='ok'):
 return {'returned_token_ids':tokens,'final':{'stop_type':stop,'truncated':truncated,'tokens_predicted':len(tokens)},'raw_text':text,'saw_stop':True,'final_token_count':0,'malformed_sse_frames':0}
class TestCap(unittest.TestCase):
 def test_profile_transport_request_same_cap(self):
  self.assertEqual(native_identity.PROFILE['output'],nt.COMPLETION_CAP)
  self.assertEqual(native_identity.PROFILE['model_profile']['maxOutputTokens'],nt.COMPLETION_CAP)
  self.assertEqual(nt.make_completion_payload([0,9])['n_predict'],nt.COMPLETION_CAP)
 def test_allowed_exact_values_only(self):
  self.assertIn(nt.COMPLETION_CAP,cap_contract.ALLOWED_OUTPUT_CAPS)
  for bad in (0,191,193,1024,'192',True,192.0):
   with self.assertRaises(ValueError):cap_contract.validate_cap(bad)
 def test_profile_mismatch_rejected(self):
  p=json.loads((ARGS.arm/'native_evaluator/profile.json').read_text());p['model_profile']['maxOutputTokens']=384 if p['output']!=384 else 768
  with tempfile.TemporaryDirectory() as td:
   q=Path(td)/'p.json';q.write_text(json.dumps(p))
   with self.assertRaisesRegex(ValueError,'mismatch'):cap_contract.load_profile(q)
 def test_early_eos_is_completion_not_cap_hit(self):
  r=nt._completion_validation(stream([8,1]),CASE,Protocol())
  self.assertTrue(r['protocol']['valid']);self.assertFalse(r['cap']['hit']);self.assertEqual(r['eos']['status'],'canonical_eos')
 def test_eos_exactly_at_inclusive_cap_is_valid(self):
  tokens=[8]*(nt.COMPLETION_CAP-1)+[1]; r=nt._completion_validation(stream(tokens),CASE,Protocol())
  self.assertEqual(r['cap']['returned_tokens_including_terminal'],nt.COMPLETION_CAP);self.assertFalse(r['cap']['hit']);self.assertTrue(r['protocol']['valid'])
 def test_limit_without_eos_is_cap_hit(self):
  r=nt._completion_validation(stream([8]*nt.COMPLETION_CAP,'limit',True),CASE,Protocol())
  self.assertTrue(r['cap']['hit']);self.assertEqual(r['cap']['status'],'hit_without_eos');self.assertFalse(r['protocol']['valid']);self.assertEqual(r['protocol']['stop_type'],'limit')
 def test_noncanonical_eog_is_not_silently_cap_hit(self):
  r=nt._completion_validation(stream([8,130073]),CASE,Protocol())
  self.assertFalse(r['cap']['hit']);self.assertEqual(r['eos']['status'],'noncanonical_native_eog');self.assertFalse(r['protocol']['valid'])
if __name__=='__main__':unittest.main()
