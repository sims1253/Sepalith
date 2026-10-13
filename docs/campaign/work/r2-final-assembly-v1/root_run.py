"""Bounded clean-environment CLI launcher. Root admission is required inside child."""
import os,subprocess,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent

def command(arguments):
    if '--observe-synthetic-only' in arguments:raise ValueError('use observe_cpu.py for fixed synthetic observation')
    return ['/home/m0hawk/Documents/Sepalith/.venv/bin/python','-I','-S','-B',str(HERE/'assemble_final.py'),*arguments]

def main():
    env={k:v for k,v in os.environ.items() if not k.startswith(('LD_','PYTHON','GGML_'))}
    env.update(CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_DATASETS_OFFLINE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false')
    try:return subprocess.run(command(sys.argv[1:]),env=env,timeout=1200,check=False).returncode
    except subprocess.TimeoutExpired:
        print('assembly wrapper exceeded the 1200-second CPU deadline',file=sys.stderr);return 124
if __name__=='__main__':raise SystemExit(main())
