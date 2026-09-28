"""Separate fixed DEV entry point; does not call or override a final gate."""
import argparse, json, os, sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
sys.path.insert(0,'/home/m0hawk/Documents/Sepalith/.venv-sft/lib/python3.10/site-packages')
import native_evaluator as evaluator
import native_transport as transport
from native_identity import NativeOwner
import final_binding as binding
import final_row_gate as gate
PANEL=Path('/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb/docs/campaign/work/lead/corrected-dev75-v1/dev75-corrected-finish-v1.jsonl')
PANEL_SHA='7c144bcd20570b1b1b1a232a5d90589c281ac2955d5fc2ef4b4b01fed8d57035'

def main():
    p=argparse.ArgumentParser();p.add_argument('--native-admission',type=Path,required=True)
    p.add_argument('--source-manifest',type=Path,required=True);p.add_argument('--source-sha256',required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--deadline-seconds',type=int,default=600)
    a=p.parse_args();os.sched_setaffinity(0,{min(os.sched_getaffinity(0))});sys.dont_write_bytecode=True
    sources=json.loads(a.source_manifest.read_text());admission=json.loads(a.native_admission.read_text())
    if admission.get('purpose')!='dev-rehearsal':raise ValueError('explicit DEV-only native admission required')
    def verify_sources():binding.verify_source_closure(sources,a.source_sha256)
    verify_sources()
    if transport.sha256_file(PANEL)!=PANEL_SHA:raise ValueError('fixed corrected DEV bytes changed')
    rows=[json.loads(line) for line in PANEL.read_text().splitlines()]
    if len(rows)!=75 or len({r['id'] for r in rows})!=75 or any(r['split']!='dev' for r in rows) or sum(r['operation']=='no_op' for r in rows)!=32:raise ValueError('exact DEV75 required')
    owner=NativeOwner(admission);owner.acquire()
    try:
        return evaluator.evaluate_rows(rows,gate._load_protocol(),owner,sources,{'route':'DEV-only separate rehearsal',
            'panel_sha256':PANEL_SHA,'source_closure_sha256':a.source_sha256,'final_access':False},
            a.output,evaluator.PROFILE['selection']['tokenizer_dir'],
            a.deadline_seconds,lambda:None,verify_sources)
    finally:owner.close()

if __name__=='__main__':main()
