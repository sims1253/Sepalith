#!/usr/bin/env python3
import hashlib,json
from pathlib import Path
PLAN=Path(__file__).resolve().parents[5]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def verify(root,name):
 m=json.loads((root/name).read_text())
 for item in m['files'] if 'files' in m else m['source_files']:
  p=root/item['path'];assert p.is_file(),p;assert sha(p)==item['sha256'],p;assert p.stat().st_size==item['bytes'],p
 return sha(root/name)
h=PLAN/'docs/campaign/work/lead/r2-expanded-provider-abortable-helper-preparation-v1';s=PLAN/'docs/campaign/work/lead/r2-expanded-provider-serving-parity-preparation-v1'
assert verify(h,'source-manifest.json')=='f3573fa68281ea9552b1c9560491fa43a85567a3bb1745eab2955406dbf1ae8c'
assert verify(h,'artifact-manifest.json')=='a13293dd08714d61a5f3ca328fef1956e344138c1d4399753ab1099cf7e51310'
assert sha(s/'source-manifest.json')=='b76c998e4f027bc5c7da9c62a53df6573dd862009901e3805300b2417f28ade5'
print('PASS exact input manifests and predecessor pin')
