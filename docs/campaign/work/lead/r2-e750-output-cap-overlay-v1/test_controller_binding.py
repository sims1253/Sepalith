#!/usr/bin/env python3
import argparse,copy,datetime,json,sys,unittest
from pathlib import Path
ap=argparse.ArgumentParser(add_help=False);ap.add_argument('--arm',type=Path,required=True);args,rest=ap.parse_known_args();sys.argv=[sys.argv[0],*rest]
sys.path[:0]=[str(args.arm/'native_controller'),str(args.arm/'native_evaluator')]
import root_controller
class TestControllerBinding(unittest.TestCase):
 def admission(self):
  d=json.loads((args.arm/'root-admission.template.json').read_text());now=datetime.datetime.now(datetime.timezone.utc);d.update(status='root_admitted',run_id='unit-test',lease_id='unit-test',created_at=now.isoformat(),expires_at=(now+datetime.timedelta(minutes=30)).isoformat());return d
 def test_exact_cap_profile_accepted(self):
  d=root_controller.validate_approval(self.admission());self.assertEqual(d['output_cap'],root_controller.identity.PROFILE['output'])
 def test_cap_swap_rejected(self):
  d=self.admission();d['output_cap']={192:384,384:768,768:192}[d['output_cap']]
  with self.assertRaisesRegex(ValueError,'cap/profile'):root_controller.validate_approval(d)
 def test_profile_hash_swap_rejected(self):
  d=self.admission();d['profile_sha256']='0'*64
  with self.assertRaisesRegex(ValueError,'profile'):root_controller.validate_approval(d)
if __name__=='__main__':unittest.main()
