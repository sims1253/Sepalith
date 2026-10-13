#!/usr/bin/env python3
"""Create the exact E750 native profile from completed root metadata."""
import json,sys
from pathlib import Path
H=Path(__file__).resolve().parent;sys.path[:0]=[str(H/'packet'),str(H/'packet/native_evaluator')]
from bind_native_profile import build,load,sha
P=Path('/mnt/e/sepalith/campaign-20260915/models/SFT11-expanded-postsft500-e-750-merged/parent-manifest.preparation.json')
I=H/'root-bindings/export-integrity.json';O=H/'packet/native_evaluator/profile.json'
def main():
 ps,ins=sha(P),sha(I);profile=build(load(P,ps),load(I,ins),ps,ins)
 if O.exists():raise ValueError('fresh profile path required')
 O.write_text(json.dumps(profile,indent=2)+'\n');print(json.dumps({'profile':str(O),'sha256':sha(O),'root_native_admission_still_required':True}))
if __name__=='__main__':main()
