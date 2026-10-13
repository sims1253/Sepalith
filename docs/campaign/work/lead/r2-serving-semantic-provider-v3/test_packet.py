#!/usr/bin/env python3
import hashlib,json,re
from pathlib import Path
P=Path(__file__).parent
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
base=json.loads((P/'r-base-bindings.json').read_text())
assert base['count']==1407 and base['names_lf_sha256']=='47715cd060c0dab5b657c9732bc0e103eda3e6d889d77c44d219489021361a44'
ts=(P/'source/r_base_bindings.generated.ts').read_text()
assert base['runtime'] in ts and base['names_lf_sha256'] in ts
helper=json.loads((P/'actual-helper-screen.json').read_text())['rows']
assert len(helper)==8 and sum(r['exact_helper_names'] for r in helper)==2
c=json.loads((P/'census-screen.json').read_text())
assert c['denominator']==128 and c['analyzer']=={'supported':49,'holds':79,'supported_with_helpers':8}
assert c['provider_on_analyzer_supported']['supported']==30 and c['provider_on_analyzer_supported']['unresolved']==19
assert c['provider_on_analyzer_supported']['helper_exact']==2 and c['provider_on_analyzer_supported']['helper_denominator']==8
for name in ['actual-helper-screen.json','census-screen.json']:
    remote=json.loads((P/'notebook'/name).read_text())
    local=json.loads((P/name).read_text())
    # Runtime timings differ; row-level evidence must be exact.
    if name=='census-screen.json': remote.pop('elapsed_ms',None); local.pop('elapsed_ms',None)
    assert remote==local
assert not (P/'notebook/processes-after.txt').read_text()
assert (P/'notebook/helper.exit').read_text().strip()=='0' and (P/'notebook/census.exit').read_text().strip()=='0'
manifest=json.loads((P/'artifact-manifest.json').read_text())
for f in manifest['files']:
    path=P/f['path']; assert path.stat().st_size==f['bytes'] and sha(path)==f['sha256']
print('PASS packet hashes, base metadata, 8-row parity, 128-row census, notebook cleanup')
