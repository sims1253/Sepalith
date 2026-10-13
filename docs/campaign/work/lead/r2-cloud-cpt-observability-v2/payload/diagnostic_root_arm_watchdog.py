#!/usr/bin/env python3
"""Validate and arm the 900-second bootstrap-diagnostic watchdog."""
import argparse,json,os,sys
from pathlib import Path
from diagnostic_contract import require,sha,validate
WATCHDOG=Path(__file__).with_name('deadline_watchdog.py');EXPECTED='fbbe410de51f786a08b0647c3a09a8c816aede76445656d27e57daccd2406885'
def main():
 parser=argparse.ArgumentParser();parser.add_argument('binding',type=Path);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();binding=json.loads(args.binding.read_text());validate(dict(binding,watchdog_armed_receipt_sha256='0'*64));require(sha(WATCHDOG)==EXPECTED,'watchdog source differs')
 os.execv(sys.executable,[sys.executable,'-B',str(WATCHDOG),'--name','sepalith-cpt-'+binding['run_id'],'--deadline-utc',binding['absolute_deadline_utc'],'--output',str(args.output)])
if __name__=='__main__':main()
