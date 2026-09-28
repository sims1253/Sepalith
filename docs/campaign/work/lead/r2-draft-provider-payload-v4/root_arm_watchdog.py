"""Arm the independent draft watchdog only for the exact reviewed binding."""
import argparse,json,os,sys,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'payload'))
from cloud_entry import validate_binding
EXPECTED="caa22ad5dc4871562d0df5f05e0f36a981ecf25cbd670e188b3d8d9c4e58ddc7"
def main():
 p=argparse.ArgumentParser();p.add_argument('binding',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 b=json.loads(a.binding.read_text());validate_binding(dict(b,watchdog_armed_receipt_sha256='0'*64))
 watchdog=ROOT/'deadline_watchdog.py'
 assert hashlib.sha256(watchdog.read_bytes()).hexdigest()==EXPECTED
 os.execv(sys.executable,[sys.executable,'-B',str(watchdog),'--name','sepalith-r2-draft-'+b['run_id'],'--deadline-utc',b['absolute_deadline_utc'],'--output',str(a.output)])
if __name__=='__main__':main()
