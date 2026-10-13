"""Root-only configuration check around the existing independent watchdog."""
import argparse,json,os,sys
from pathlib import Path
from cloud_entry import validate_binding,sha,require
WATCHDOG=Path(__file__).with_name('deadline_watchdog.py')
EXPECTED='0e1a2d8a02cd4ed5807dc9cc0ddcc90d64e5c4dda4b9d28a211b9ed67ca65721'
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
