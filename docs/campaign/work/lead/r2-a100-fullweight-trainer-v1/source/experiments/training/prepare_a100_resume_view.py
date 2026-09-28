#!/usr/bin/env python3
"""Create the immutable-source local-to-DDP resume view after root admission."""
import argparse, json
from pathlib import Path
from a100_distributed import create_initial_resume_view

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--source",type=Path,required=True)
    p.add_argument("--destination",type=Path,required=True)
    p.add_argument("--manifest-sha256",required=True)
    p.add_argument("--world-size",type=int,default=8)
    a=p.parse_args()
    print(json.dumps(create_initial_resume_view(a.source,a.destination,
        manifest_sha256=a.manifest_sha256,world_size=a.world_size),sort_keys=True))
if __name__=="__main__": main()
