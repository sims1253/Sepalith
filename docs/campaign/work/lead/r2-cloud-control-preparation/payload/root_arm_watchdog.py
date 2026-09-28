"""Root-only configuration check around the existing independent watchdog."""
import argparse,json,os,sys
from pathlib import Path
from cloud_entry import validate_binding,sha,require
WATCHDOG=Path('/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb/docs/campaign/work/r2-cloud-control-independent-review-v1/deadline_watchdog_delayed_submission.py')
EXPECTED='16d3e01bdb0f73bfa92574804a3af7feaad1e722bd87782a1a61c408c5b29514'
def main():
 p=argparse.ArgumentParser();p.add_argument('binding',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 b=json.loads(a.binding.read_text())
 # The armed receipt is created by the watchdog we are about to exec. Only
 # this arming check skips that unavailable hash; cloud entry still requires
 # root to bind the real receipt after the watchdog confirms it is armed.
 arming=dict(b,watchdog_armed_receipt_sha256='0'*64);validate_binding(arming)
 require(sha(WATCHDOG)==EXPECTED,'independent watchdog source differs')
 # validate_binding enforces deadline <= armed_at+7200. The general reused
 # watchdog has a six-hour validator ceiling; this root wrapper narrows it.
 os.execv(sys.executable,[sys.executable,'-B',str(WATCHDOG),'--name','sepalith-r2-control-'+b['run_id'],'--deadline-utc',b['absolute_deadline_utc'],'--output',str(a.output)])
if __name__=='__main__':main()
