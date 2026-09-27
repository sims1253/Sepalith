#!/usr/bin/env python3
import hashlib,json,subprocess
from pathlib import Path
HERE=Path(__file__).resolve().parent
UP=Path('/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb/docs/campaign/work/lead/r2-semantic10948-provider-materialization-v2')
binding=json.loads((UP/'provider-binding.json').read_text())
for item in binding['verified_files']:
 p=HERE/item['path'];assert p.is_file() and p.stat().st_size==item['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==item['sha256'],p
assert binding['verified_file_count']==24
template=json.loads((HERE/'dedup-binding.template.json').read_text());assert template['status'].startswith('unbound_') and template['training_admission'] is False
source=(HERE/'materialize9535.py').read_text();assert "status']=='root_bound_after_semantic10948_finalized'" in source and "dedup_against_current20191_and_finalized10948" in source
assert "except (B.Hold, ValueError, UnicodeDecodeError)" in (HERE/'prepare_shard_base.py').read_text()
assert "except Exception as e:holds.write" not in (HERE/'prepare_shard_base.py').read_text()
for script in ('prepare_inputs.py','materialize9535.py'):
 r=subprocess.run(['python3','-B',str(HERE/script),'--help'],capture_output=True,text=True);assert r.returncode==0,(script,r.stderr)
print('PASS exact 24-file provider closure, fail-closed infrastructure policy, unbound dedup gate, CLI entrypoints')
