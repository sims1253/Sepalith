#!/usr/bin/env python3
"""Apply the reviewed native allocator policy and preserve the root CUDA lease."""
import hashlib,json,os,runpy,sys
from pathlib import Path
CONFIG='backend:native,roundup_power2_divisions:[32:256,64:128,256:64,>:32],garbage_collection_threshold:0.8'
RUNNER=Path('/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb/docs/campaign/work/lead/r2-native-varlen-canary-preparation-v1/source/experiments/training/native_varlen_canary.py')
EXPECTED='fc9516057d9626b502221765fe6f43a58a23c4f1e7b21a4e0d976964715b6437'
def main():
 if hashlib.sha256(RUNNER.read_bytes()).hexdigest()!=EXPECTED:raise RuntimeError('Frozen canary runner differs')
 fd=int(os.environ['SEPALITH_CUDA_LOCK_FD']);actual=os.fstat(fd);expected=os.stat('/home/m0hawk/.local/state/sepalith/campaign-20260915/resource-locks/cuda0.lock')
 if (actual.st_dev,actual.st_ino)!=(expected.st_dev,expected.st_ino):raise RuntimeError('Inherited CUDA lease inode differs')
 if os.environ.get('PYTORCH_ALLOC_CONF')!=CONFIG:raise RuntimeError('Allocator policy differs')
 import unsloth,torch
 if torch.cuda.memory.get_allocator_backend()!='native':raise RuntimeError('Allocator backend differs')
 torch.cuda.memory.set_per_process_memory_fraction(.95,device=0)
 report=Path(os.environ['SEPALITH_ALLOCATOR_REPORT']);report.parent.mkdir(parents=True,exist_ok=True)
 if report.exists():raise RuntimeError('Allocator report already exists')
 report.write_text(json.dumps({'schema':'sepalith.sft11.native-varlen-canary-allocator.v1','runner_sha256':EXPECTED,'backend':'native','memory_fraction':.95},sort_keys=True)+'\n')
 sys.path.insert(0,str(RUNNER.parent));sys.argv[0]=str(RUNNER);runpy.run_path(str(RUNNER),run_name='__main__')
if __name__=='__main__':main()
