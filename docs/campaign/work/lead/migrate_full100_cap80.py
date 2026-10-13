"""Create an explicit resource-cap derivative; preserve the original checkpoint."""
from pathlib import Path
import copy
import datetime as dt
import hashlib
import json
import shutil
import sys

PLAN = Path('/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb')
BASE = Path('/home/m0hawk/.local/state/sepalith/campaign-20260915')
SNAPSHOT = Path('/home/m0hawk/.local/state/sepalith/migration-20260906/prepared-state/snapshots/be406557c23d368fbb9869c0edf0c0c1099c7939d650a6bc32190fce53c330b9/source')
sys.path.insert(0, str(SNAPSHOT / 'experiments/training'))
import campaign_checkpoint as cc

source = BASE / 'training/RL-primary-p2-mb4-full5-c/archive/full/checkpoint-100'
root = BASE / 'checkpoints/RL-primary-full100-cap80/full'
destination = root / 'checkpoint-100'
staging = root / '.checkpoint-100.staging'
receipt_path = PLAN / 'docs/campaign/receipts/RL-08-full100-cap80-migration.json'


def require(value, message):
    if not value:
        raise RuntimeError(message)


def identity_hash(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


require(not destination.exists() and not staging.exists() and not receipt_path.exists(),
        'Fresh derivative and receipt required')
old_manifest_hash = cc.digest(source / 'campaign-manifest.json')
old_state_hash = cc.digest(source / 'campaign-state.json')
require(old_manifest_hash == '2fa678487c5a760a2d04b798fc29787cd300752e7b6bd3445fdb8a9b09399ebd',
        'Original manifest hash mismatch')
require(old_state_hash == '930385c7715c91d3be3e9a570ee8de752d12f04ef1e6db55b10aab05900b9ed8',
        'Original state hash mismatch')
original = cc.verify_checkpoint(source, require_full=True)
require(original['step'] == 100, 'Expected full100')
old_identity_hash = identity_hash(original['identity'])
require(old_identity_hash == '48028a843276d3ebe32b0b07fe9beb77e465dfc8aef920586eb468a8dbf9a0f2',
        'Original identity mismatch')
require(original['identity']['policy']['cuda_memory_fraction'] == .75, 'Expected75% cap')
new_identity = copy.deepcopy(original['identity'])
new_identity['policy']['cuda_memory_fraction'] = .80
check = copy.deepcopy(new_identity)
check['policy']['cuda_memory_fraction'] = .75
require(check == original['identity'], 'Only allocator cap may differ')
del check
state = json.loads((source / 'campaign-state.json').read_text())
require(state['identity'] == original['identity'] and state['step'] == 100 and state['full'],
        'State/manifest identity mismatch')
sampler = state['sampler']
require(sampler['source_draw_cursor'] == 800, 'Expected source cursor800')
del state
new_identity_hash = identity_hash(new_identity)
require(new_identity_hash == 'ebf964b456eff1c2df2b3064277afe346db9a358eb47b5a902a203822b5b4818', 'Derived identity mismatch')
root.mkdir(parents=True, exist_ok=True)
staging.mkdir()
preserved = {}
for name, info in original['files'].items():
    if name == 'campaign-state.json':
        continue
    relative = Path(name)
    require(not relative.is_absolute() and '..' not in relative.parts, 'Invalid file path')
    target = staging / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source / relative, target)
    require(cc.digest(target) == info['sha256'] and target.stat().st_size == info['bytes'],
            f'Copied file mismatch: {name}')
    preserved[name] = info
provenance = {
    'schema_version': 'sepalith.resource-cap-checkpoint-derivative.v1',
    'at': dt.datetime.now(dt.timezone.utc).isoformat(),
    'source_checkpoint': str(source), 'source_manifest_sha256': old_manifest_hash,
    'source_state_sha256': old_state_hash, 'source_identity_sha256': old_identity_hash,
    'derived_identity_sha256': new_identity_hash,
    'identity_change': {'path': 'policy.cuda_memory_fraction', 'before': .75, 'after': .80},
    'reason': 'Update104 allocation exceeded75% allocator allowance with physicalVRAM available;80% is within the unchanged source validator limit.',
    'preserved_files': preserved,
    'sampler_source_cursor': 800,
    'scope': 'Resource identity derivative only. Adapter/optimizer/scheduler/RNG/trainer/tokenizer bytes and sampler state unchanged. Original checkpoint preserved.',
    'acceptance': 'Not a new model checkpoint or quality result. Actual cap80 replay and update104 remain to be tested.',
}
cc.write_json(staging / 'resource-cap-migration.json', provenance)
derived = cc.seal_checkpoint(staging, new_identity, 100, full=True, sampler=sampler)
cc.verify_checkpoint(staging, new_identity, require_full=True)
derived_state = json.loads((staging / 'campaign-state.json').read_text())
require(derived_state['sampler'] == sampler, 'Sampler changed')
require(derived_state['identity'] == new_identity, 'Derived state identity mismatch')
for name, info in preserved.items():
    require(derived['files'][name] == info, f'Mathematical state changed: {name}')
require(cc.digest(source / 'campaign-manifest.json') == old_manifest_hash and
        cc.digest(source / 'campaign-state.json') == old_state_hash,
        'Original metadata changed')
staging.rename(destination)
cc.flush_directory(root)
receipt = {
    'task': 'RL-08', 'owner': 'lead', 'at': dt.datetime.now(dt.timezone.utc).isoformat(),
    'status': 'explicit_resource_cap_derivative_verified_not_launched',
    'source': str(source), 'destination': str(destination),
    'source_manifest_sha256': old_manifest_hash, 'source_state_sha256': old_state_hash,
    'source_identity_sha256': old_identity_hash, 'derived_identity_sha256': new_identity_hash,
    'derived_manifest_sha256': cc.digest(destination / 'campaign-manifest.json'),
    'derived_state_sha256': cc.digest(destination / 'campaign-state.json'),
    'migration_provenance_sha256': cc.digest(destination / 'resource-cap-migration.json'),
    'preserved_files': preserved, 'sampler_identical': True,
    'strict_production_seal_and_verify': True,
    'source_snapshot_unchanged': str(SNAPSHOT),
    'math_and_quality': 'No mathematical state, model weight or evaluation change claimed. Runtime fraction alone differs; no resume equivalence claim until measured.',
    'next_gate': 'Root reviews derivative, prepares matching80% recipe and verifies actual first3 replay updates plus formerly failing104.',
}
cc.write_json(receipt_path, receipt)
print(json.dumps({k: receipt[k] for k in ('status', 'destination', 'derived_identity_sha256', 'derived_manifest_sha256')}))
