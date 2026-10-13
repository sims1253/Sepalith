#!/usr/bin/env python3
import argparse
from pathlib import Path
from return_checkpoint_common import build
from transport_common import atomic_json

def main():
 p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--accepted-manifest-sha256',required=True);p.add_argument('--prefix',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();atomic_json(a.output,build(a.checkpoint,a.accepted_manifest_sha256,'scholzmx/sepalith-lora',a.prefix))
if __name__=='__main__':main()
