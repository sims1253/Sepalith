#!/usr/bin/env python3
import hashlib,json
from pathlib import Path
p=Path(__file__).resolve().parents[1]
def sha(q):return hashlib.sha256(Path(q).read_bytes()).hexdigest()
m=json.loads((p/'selected-source-manifest.json').read_text());assert m['records_count']==5
for item in m['records']:
 local=p/item['path'];original=Path(item['original_path']);assert local.read_bytes()==original.read_bytes();assert local.stat().st_size==item['bytes'];assert sha(local)==item['sha256']
metric=json.loads((p/'selected-source/metric-selection-result.json').read_text());assert metric['status']=='matched_development_metrics_root_verified';assert metric['final_set_opened'] is False
for panel in ('anchor2k','8k','16k'):assert metric['arms']['varlen_candidate']['panels'][panel]['mean_causal_nll'] < metric['arms']['ordinary_reference']['panels'][panel]['mean_causal_nll']
payload=json.loads((p/'selected-source/payload-review.json').read_text())['arms']['varlen_candidate'];assert payload['native_and_durable_hashes_verified'] is True and payload['files_verified_each_copy']==12 and payload['checkpoint_manifest_sha256']=='2b35854537588499fb703af2127029ff649e410cbab65374948723273b520951'
recipe=json.loads((p/'selected-source/canary-runtime-recipe.json').read_text());v=recipe['varlen_canary'];assert v['arm']=='varlen_candidate' and v['updates']==8 and v['source_global_step']==322 and v['target_global_step']==330 and v['source_cursor']==4096 and v['target_cursor']==4224;assert v['physical_execution']['trainer_gradient_accumulation']==1
print('PASS selected packed330 recipe, 3-panel metric decision, and 12-file root payload review pins')
