"""Prepare a same-identity full-state continuation; no CUDA launch."""
from pathlib import Path
import datetime as dt
import hashlib
import json
import shutil
import sys

P = Path('/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb')
C = P / 'docs/campaign'
N = Path('/home/m0hawk/.local/state/sepalith/campaign-20260915')
oldrun = N / 'training/RL-primary-p2-mb4-full5-a'
run = N / 'training/RL-primary-p2-mb4-full5-b'
oldpath = C / 'work/main-rl-preparation/primary-mb4-full5-a.recipe.json'

def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()

def write(p, value):
    with p.open('x') as f:
        json.dump(value, f, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        f.write('\n')

assert sha(oldpath) == 'c85b7e3e9fd7927d429a6d45085e98d3f5bb8b06047e468f5bd7dc9190939969'
r = json.loads(oldpath.read_text())
identity_sha = hashlib.sha256(json.dumps(r['identity'], ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
assert identity_sha == '48028a843276d3ebe32b0b07fe9beb77e465dfc8aef920586eb468a8dbf9a0f2'
terminal = json.loads((oldrun.with_name(oldrun.name+'-host-supervision')/'terminal.json').read_text())
assert terminal['status'] == 'completed' and terminal['child_exit_code'] == 0
devpath = oldrun / 'archive/evaluations/cases-step-5.json'
dev = json.loads(devpath.read_text())
assert dev['status'] == 'complete' and dev['step'] == 5
assert dev['summary']['counts'] == {'cap_hit':7,'edit_exact':26,'exact_region':51,'predicted_noop':26,'protocol_valid':68,'strict_noop_correct':25,'strict_noop_false_suggestions':5,'suggestion':42}
assert len(dev['results']) == 75 and len({x['id'] for x in dev['results']}) == 75
assert [x['id'] for x in dev['results']] == r['development_case_ids']
assert sum(x['expected_noop'] for x in dev['results']) == 32
assert not run.exists()
source = Path(r['identity']['source']['frozen_source_root'])
sys.path[:0] = [str(source/'experiments/training'), str(source/'packages/sepalith/src')]
from campaign_rl_entry import preflight_entry

r.update(id='primary-grpo-p2-2048x192-v5-full5-b', output_dir=str(run/'output'),
         archive_root=str(run/'archive'), telemetry_path=str(run/'telemetry.jsonl'),
         resume_from=str(oldrun/'archive/full/checkpoint-5'), decision_steps=[25])
recipepath = oldpath.with_name('primary-mb4-full5-b.recipe.json')
write(recipepath, r)
audit = preflight_entry(r)
assert audit['status'] == 'preflight_pass'
fullbytes = sum(x.stat().st_size for x in (oldrun/'archive/full/checkpoint-5').iterdir() if x.is_file())
lightbytes = sum(x.stat().st_size for x in (oldrun/'archive/adapters/checkpoint-5').iterdir() if x.is_file())
nativefree = shutil.disk_usage(N).free
# Output and archive copies plus light checkpoints, four new boundaries.
forecast_next20 = 4 * (2*fullbytes+lightbytes) + 2_000_000_000
assert nativefree-forecast_next20 >= 100_000_000_000
command = json.loads(oldpath.with_name('primary-mb4-full5-a.command.json').read_text())
command[command.index(str(oldpath))] = str(recipepath)
command[-1] = str(run/'supervision.json')
commandpath = recipepath.with_name('primary-mb4-full5-b.command.json')
write(commandpath, command)
charged = 3748.6021434420254 + terminal['seconds']
packet = {'task':'RL-07/RL-08','owner':'lead','at':dt.datetime.now(dt.timezone.utc).isoformat(),
          'status':'prepared_preflight_pass_pending_resource_admission',
          'recipe':str(recipepath),'recipe_sha256':sha(recipepath),'identity_sha256':identity_sha,
          'source_snapshot':source.parent.name,'resume_from':r['resume_from'],
          'resume_manifest_sha256':sha(Path(r['resume_from'])/'campaign-manifest.json'),
          'preflight_status':audit['status'],'resume_audit':audit['resume_audit'],
          'prior_terminal':terminal,'DEV_summary':{k:v for k,v in dev['summary'].items() if k != 'case_ids'},
          'DEV_artifact_sha256':sha(devpath),
          'decision':'Continue to25: first5 matches theta0 on edits/noops/FP/protocol/caps. Too early to judge RL; no checkpoint promotion.',
          'same_identity':True,'changed_attempt_fields':['id','output_dir','archive_root','telemetry_path','resume_from','decision_steps'],
          'budget':{'ceiling_seconds':93600,'charged_before_seconds':charged,'remaining_before_seconds':93600-charged,'attempt_seconds':5400,'guard_seconds':5460,'cloud_delta':0,'hard_stop':'2026-09-13T22:00:00Z'},
          'storage':{'free_bytes':nativefree,'full_checkpoint_bytes':fullbytes,'light_checkpoint_bytes':lightbytes,'next20_upper_estimate_bytes':forecast_next20,'free_floor_bytes':100_000_000_000,'policy':'Retain all existing artifacts. This bounded20-update continuation fits. Reassess cadence/retention before a longer allocation;3000 remains an unallocated ceiling.'},
          'acceptance':['Actual resume cursor40 and next8source draws; finite update6; full state restored', 'Full25 and exact DEV75 before next continuation decision'],
          'displaced_work':'Continues primary RL within existing26h ceiling. Optional kernel sweeps remain cut.'}
write(C/'receipts/RL-07-full5-resume25-preparation.json',packet)
print(json.dumps({k:v for k,v in packet.items() if k not in ['DEV_summary','resume_audit']},indent=2))
