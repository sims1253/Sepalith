"""Additive source guard around the unchanged, separate DEV-only entry point."""
import hashlib, json, os, sys, time
from pathlib import Path
HERE=Path(__file__).resolve().parent
CANDIDATE=HERE.parent/'native_evaluator'
SITE='/home/m0hawk/Documents/Sepalith/.venv-sft/lib/python3.10/site-packages'
sys.dont_write_bytecode=True
sys.path[:0]=[str(HERE),str(CANDIDATE),SITE]
import origin_snapshot
import rehearse_dev as dev
import client_source_policy as policy

def main():
    os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
    if not sys.flags.isolated or not sys.flags.no_site:raise ValueError('explicit -I -S required')
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CPU tokenizer client requires CUDA hidden')
    # AutoTokenizer reads only the original tokenizer/config. No model class/load.
    tokenizer,audit=dev.transport._load_tokenizer(Path(dev.evaluator.PROFILE['selection']['tokenizer_dir']))
    ids=tokenizer.encode('x <- 1\n',add_special_tokens=False,split_special_tokens=True)
    tokenizer.decode(ids,skip_special_tokens=False,clean_up_tokenization_spaces=False)
    # The unchanged evaluator creates its own tokenizer. Release this observed
    # preload before that call to avoid retaining two tokenizer objects.
    del tokenizer
    dev.gate._load_protocol()
    import encodings.idna, http.client
    dev.transport.Request('http://127.0.0.1:18403/props')
    if sys.argv[1:]==['--check-policy-only']:
        graph=json.loads((HERE/'source-closure.prepared.json').read_text())
        print(json.dumps(policy.verify(graph,deep=True)))
        return
    # Record this exact entry point, imports, tokenizer and pure synthetic text.
    if sys.argv[1:] in (['--observe-only'],['--observe-and-check']):
        observation=origin_snapshot.snapshot()
        observation['executable_maps']=policy.executable_maps()
        observation['tokenizer']=audit;observation['synthetic_ids']=ids
        with (HERE/'client-origins.json').open('x') as f:json.dump(observation,f,indent=2)
        if sys.argv[1:]==['--observe-and-check']:
            deadline=time.monotonic()+60
            while not (HERE/'source-closure.prepared.json').exists():
                if time.monotonic()>deadline:raise TimeoutError('CPU graph preparation handshake expired')
                time.sleep(.05)
            graph=json.loads((HERE/'source-closure.prepared.json').read_text())
            print(json.dumps(policy.verify(graph,deep=True)))
        return
    manifest_arg=sys.argv.index('--source-manifest')+1
    graph=json.loads(Path(sys.argv[manifest_arg]).read_text())
    original_closure=dev.binding.verify_source_closure
    original_native=dev.NativeOwner.verify
    def additional_closure(manifest,expected):
        result=original_closure(manifest,expected)
        policy.verify(manifest,deep=True)
        return result
    def additional_native(owner,**kwargs):
        policy.verify(graph,deep=False)
        return original_native(owner,**kwargs)
    # These wrappers add origin checks; all original source/native/time/row gates run.
    dev.binding.verify_source_closure=additional_closure
    dev.NativeOwner.verify=additional_native
    return dev.main()

if __name__=='__main__':main()
