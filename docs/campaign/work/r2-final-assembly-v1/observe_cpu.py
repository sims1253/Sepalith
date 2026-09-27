"""Bounded CPU synthetic origin observation; no final rows or native server."""
import argparse,json,os,signal,subprocess,time
from pathlib import Path
HERE=Path(__file__).resolve().parent
def main():
    p=argparse.ArgumentParser();p.add_argument('route',choices=('assembly',));a=p.parse_args()
    os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
    python='/home/m0hawk/Documents/Sepalith/'+'.venv'+'/bin/python'
    entry='assemble_final.py'
    env={k:v for k,v in os.environ.items() if not k.startswith(('LD_','PYTHON','GGML_'))}
    env.update(CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_DATASETS_OFFLINE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false')
    start=time.monotonic();peak=0;failure=None
    with (HERE/(a.route+'-observation.log')).open('xb') as log:
        child=subprocess.Popen([python,'-I','-S','-B',str(HERE/entry),'--observe-synthetic-only'],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:
            while child.poll() is None:
                try:rss=int(next(x for x in Path(f'/proc/{child.pid}/status').read_text().splitlines() if x.startswith('VmRSS:')).split()[1])*1024
                except (FileNotFoundError,StopIteration):rss=0
                peak=max(peak,rss)
                if rss>1536*1024**2:failure='1.5GiB';break
                if time.monotonic()-start>90:failure='90seconds';break
                time.sleep(.1)
        finally:
            if child.poll() is None:os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=5)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
    result={'route':a.route,'exit':child.returncode,'failure':failure,'seconds':time.monotonic()-start,'peak_rss_bytes':peak,'model_load':False,'native_launch':False,'final_access':False}
    (HERE/(a.route+'-observation-terminal.json')).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
    if failure or child.returncode:raise SystemExit(1)
if __name__=='__main__':main()
