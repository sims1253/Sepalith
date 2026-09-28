"""Prepare the next bounded same-identity continuation without loading CUDA."""
from pathlib import Path
import datetime as dt
import hashlib
import json
import shutil
import sys

P = Path('/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb')
C = P / 'docs/campaign'
N = Path('/home/m0hawk/.local/state/sepalith/campaign-20260915')
oldrun = N / 'training/RL-primary-p2-mb4-full5-b'
run = N / 'training/RL-primary-p2-mb4-full5-c'
oldpath = C / 'work/main-rl-preparation/primary-mb4-full5-b.recipe.json'

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

assert sha(oldpath) == 'ea27572c81d46c7a36a54a5ed05def7fb9decd044abec103586d5bd2cbcde56d'
r = json.loads(oldpath.read_text())
identity_sha = hashlib.sha256(json.dumps(r['identity'], ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
assert identity_sha == '48028a843276d3ebe32b0b07fe9beb77e465dfc8aef920586eb468a8dbf9a0f2'
terminal = json.loads((oldrun.with_name(oldrun.name+'-host-supervision')/'terminal.json').read_text())
assert terminal['status'] == 'completed' and terminal['child_exit_code'] == 0
devpath = oldrun / 'archive/evaluations/cases-step-25.json'
dev = json.loads(devpath.read_text())
assert dev['status'] == 'complete' and dev['step'] == 25
assert dev['summary']['counts'] == {'cap_hit':6,'edit_exact':25,'exact_region':50,'predicted_noop':27,'protocol_valid':69,'strict_noop_correct':25,'strict_noop_false_suggestions':5,'suggestion':42}
assert len(dev['results']) == 75 and len({x['id'] for x in dev['results']}) == 75
assert [x['id'] for x in dev['results']] == r['development_case_ids']
assert sum(x['expected_noop'] for x in dev['results']) == 32
assert not run.exists()
source = Path(r['identity']['source']['frozen_source_root'])
sys.path[:0] = [str(source/'experiments/training'), str(source/'packages/sepalith/src')]
from campaign_rl_entry import preflight_entry

r.update(id='primary-grpo-p2-2048x192-v5-full5-c', output_dir=str(run/'output'),
         archive_root=str(run/'archive'), telemetry_path=str(run/'telemetry.jsonl'),
         resume_from=str(oldrun/'archive/full/checkpoint-25'), decision_steps=[100],
         max_attempt_seconds=9000)
recipepath = oldpath.with_name('primary-mb4-full5-c.recipe.json')
audit = preflight_entry(r)
assert audit['status'] == 'preflight_pass'
assert audit['resume_audit']['step'] == 25
assert audit['resume_audit']['source_draw_cursor'] == 200
fullbytes = sum(x.stat().st_size for x in (oldrun/'archive/full/checkpoint-25').iterdir() if x.is_file())
lightbytes = sum(x.stat().st_size for x in (oldrun/'archive/adapters/checkpoint-25').iterdir() if x.is_file())
nativefree = shutil.disk_usage(N).free
forecast = 15 * (2*fullbytes+lightbytes) + 3_000_000_000
assert nativefree-forecast >= 100_000_000_000
now = dt.datetime.now(dt.timezone.utc)
hard_stop = dt.datetime(2026,9,13,22,tzinfo=dt.timezone.utc)
assert (hard_stop-now).total_seconds() > 9060
write(recipepath, r)
command = json.loads(oldpath.with_name('primary-mb4-full5-b.command.json').read_text())
command[command.index(str(oldpath))] = str(recipepath)
command[-1] = str(run/'supervision.json')
commandpath = recipepath.with_name('primary-mb4-full5-c.command.json')
write(commandpath, command)
charged = 4836.651824991044 + terminal['seconds']
packet = {
    'task':'RL-08', 'owner':'lead', 'at':now.isoformat(),
    'status':'prepared_preflight_pass_pending_resource_admission',
    'recipe':str(recipepath), 'recipe_sha256':sha(recipepath),
    'identity_sha256':identity_sha, 'source_snapshot':source.parent.name,
    'resume_from':r['resume_from'],
    'resume_manifest_sha256':sha(Path(r['resume_from'])/'campaign-manifest.json'),
    'preflight_status':audit['status'], 'resume_audit':audit['resume_audit'],
    'prior_terminal':terminal,
    'DEV_summary':{k:v for k,v in dev['summary'].items() if k != 'case_ids'},
    'DEV_artifact_sha256':sha(devpath),
    'decision':'Continue provisionally to update 100, review complete DEV at 50 and 75 while training proceeds. Update 25 lost one formatting exact edit and resolved one roxygen cap/protocol failure; no-op performance unchanged. No RL promotion. A single early mixed readout is insufficient evidence to stop the primary hypothesis.',
    'same_identity':True,
    'changed_attempt_fields':['id','output_dir','archive_root','telemetry_path','resume_from','decision_steps','max_attempt_seconds'],
    'budget':{'ceiling_seconds':93600,'charged_before_seconds':charged,
              'remaining_before_seconds':93600-charged,'attempt_seconds':9000,
              'guard_seconds':9060,'cloud_delta':0,'hard_stop':hard_stop.isoformat()},
    'storage':{'free_bytes':nativefree,'full_checkpoint_bytes':fullbytes,
               'light_checkpoint_bytes':lightbytes,'next75_upper_estimate_bytes':forecast,
               'free_floor_bytes':100_000_000_000,
               'policy':'Retain existing artifacts. This 75-update allocation fits; max_steps 3000 remains an unallocated ceiling.'},
    'acceptance':['Verify resume at update 26 from source cursor 200 and finite gradients.',
                  'Review exact 75-case DEV at 50 and 75, stop on sustained deterioration or invalid checkpoint/telemetry.',
                  'Complete full100 and DEV100 before further allocation.',
                  'Stop for no useful held-out signal by update250 or six trajectory hours, whichever occurs first.'],
    'displaced_work':'Primary RL remains within the existing 26-hour ceiling. Optional local serving/kernel sweeps stay cut; notebook editor checks proceed independently.'
}
write(C/'receipts/RL-08-full25-resume100-preparation.json',packet)
print(json.dumps({k:v for k,v in packet.items() if k not in ['DEV_summary']},indent=2))
