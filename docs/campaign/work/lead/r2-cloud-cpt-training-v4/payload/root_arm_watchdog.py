"""Root-only pin and exec wrapper for the CPT watchdog."""
import argparse,json,os,sys
from pathlib import Path
from cloud_contract import require,sha,validate_binding
WATCHDOG=Path(__file__).with_name('deadline_watchdog.py')
EXPECTED='fbbe410de51f786a08b0647c3a09a8c816aede76445656d27e57daccd2406885'
def main():
 p=argparse.ArgumentParser();p.add_argument('binding',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args();b=json.loads(a.binding.read_text())
 arming=dict(b,watchdog_armed_receipt_sha256='0'*64);validate_binding(arming);require(sha(WATCHDOG)==EXPECTED,'watchdog source differs')
 os.execv(sys.executable,[sys.executable,'-B',str(WATCHDOG),'--name','sepalith-cpt-'+b['run_id'],'--deadline-utc',b['absolute_deadline_utc'],'--output',str(a.output)])
if __name__=='__main__':main()
