"""Root-only guarded native FINAL client, with actual-time gate before rows."""
import argparse,json,os,sys
from pathlib import Path
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
CANDIDATE=HERE.parent/'r2-final-evaluator-v1'
SITE='/home/m0hawk/Documents/Sepalith/.venv-sft/lib/python3.10/site-packages'
sys.path[:0]=[str(HERE),str(CANDIDATE),SITE]
import entry_gate as g
import native_evaluator as evaluator
import native_transport as transport
from native_identity import NativeOwner
import client_source_policy as policy
import origin_snapshot
TOKENIZER=Path('/home/m0hawk/.local/state/sepalith/campaign-20260915/models/SFT11-task-global-b-500-merged')

def preload():
    tokenizer,_=transport._load_tokenizer(TOKENIZER)
    ids=tokenizer.encode('x <- 1\n',add_special_tokens=False,split_special_tokens=True)
    tokenizer.decode(ids,skip_special_tokens=False,clean_up_tokenization_spaces=False)
    del tokenizer
    g.gate._load_protocol()
    import encodings.idna,http.client
    transport.Request('http://127.0.0.1:18403/props')

def execute(args):
    freeze=g.release(args.freeze,args.harness_sha256)
    graph=g.source_graph(args.source_manifest,args.source_sha256,freeze,'source_closure_sha256')
    admission=json.loads(Path(args.native_admission).read_text())
    # Native admission/profile/source freeze also precedes row-manifest access.
    evaluator.verify_native_freeze(freeze,admission,graph,args.harness_sha256)
    preload();policy.verify(graph,deep=True)
    original_source=g.binding.verify_source_closure;original_owner=NativeOwner.verify
    def source_check(manifest,digest):
        result=original_source(manifest,digest);policy.verify(manifest,deep=True);return result
    def owner_check(owner,**kwargs):
        policy.verify(graph,deep=False);return original_owner(owner,**kwargs)
    g.binding.verify_source_closure=source_check;NativeOwner.verify=owner_check
    model=None
    try:
        model=evaluator.prepare_final_native_evaluator(args.rows,
            final_rows_manifest=json.loads(Path(args.rows_manifest).read_text()),expected_manifest_sha256=args.manifest_sha256,
            freeze_receipt=freeze,expected_harness_sha256=args.harness_sha256,native_admission=admission,source_manifest=graph)
        model.evaluate(args.output,TOKENIZER,global_deadline_seconds=args.deadline_seconds)
        return 0
    finally:
        if model is not None:model.close()
        g.binding.verify_source_closure=original_source;NativeOwner.verify=original_owner

def main():
    os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
    if sys.argv[1:]==['--observe-synthetic-only']:
        preload();observed=origin_snapshot.snapshot();observed['executable_maps']=policy.executable_maps()
        with (HERE/'native-client-origins.json').open('x') as f:json.dump(observed,f,indent=2)
        print(json.dumps({'modules':len(observed['modules']),'model_load':False}));return 0
    p=argparse.ArgumentParser()
    for name in ('freeze','native-admission','source-manifest','rows-manifest','rows','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--harness-sha256',required=True);p.add_argument('--manifest-sha256',required=True);p.add_argument('--source-sha256',required=True)
    p.add_argument('--deadline-seconds',type=int,default=3600)
    return execute(p.parse_args())
if __name__=='__main__':raise SystemExit(main())
