#!/usr/bin/env python3
"""Fail-closed native-space admission for a future hot checkpoint save."""
import argparse,json,shutil
from pathlib import Path

GIB=1024**3
def require(v,m):
 if not v:raise ValueError(m)

def capacity(root,bundle_bytes,existing_hot_bytes,next_checkpoint_bytes,usage_cap_bytes=70*GIB,free_floor_bytes=70*GIB):
 root=Path(root);require(root.is_absolute() and str(root).startswith('/home/'),'hot root must be native Linux storage')
 for name,value in [('bundle',bundle_bytes),('existing hot',existing_hot_bytes),('next checkpoint',next_checkpoint_bytes)]:require(type(value)is int and value>=0,f'{name} bytes differ')
 require(type(usage_cap_bytes)is int and 0<usage_cap_bytes<=70*GIB,'native usage cap exceeds 70 GiB')
 require(type(free_floor_bytes)is int and free_floor_bytes>=70*GIB,'native free floor below 70 GiB')
 projected=bundle_bytes+existing_hot_bytes+next_checkpoint_bytes;free=shutil.disk_usage(root).free
 require(projected<=usage_cap_bytes,'projected native use exceeds cap')
 require(free-next_checkpoint_bytes>=free_floor_bytes,'next hot checkpoint violates native free floor')
 return {'schema':'sepalith.sft11.native-hot-capacity.v1','status':'admitted_capacity_only','free_before_bytes':free,'free_after_projected_bytes':free-next_checkpoint_bytes,'projected_campaign_native_bytes':projected,'usage_cap_bytes':usage_cap_bytes,'free_floor_bytes':free_floor_bytes}

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--bundle-bytes',type=int,required=True);p.add_argument('--existing-hot-bytes',type=int,required=True);p.add_argument('--next-checkpoint-bytes',type=int,required=True);a=p.parse_args();print(json.dumps(capacity(a.root,a.bundle_bytes,a.existing_hot_bytes,a.next_checkpoint_bytes),sort_keys=True));return 0
if __name__=='__main__':raise SystemExit(main())
