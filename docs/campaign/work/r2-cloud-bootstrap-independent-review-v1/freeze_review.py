"""Freeze bounded source/CPU review evidence; never inspect runtime/model artifacts."""
import datetime,hashlib,json,pathlib
H=pathlib.Path(__file__).resolve().parent;W=H.parent;R=W.parent/'receipts'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def rec(p):return {'path':str(p),'bytes':p.stat().st_size,'sha256':sha(p)}
v=json.loads((H/'verification.json').read_text())
for p in v['pins']:
 q=pathlib.Path(p['path']);assert sha(q)==p['sha256'],str(q)
for row in json.loads((W/'r2-cloud-bootstrap-recovery-v1/artifact-manifest.json').read_text())['files']:
 q=W/'r2-cloud-bootstrap-recovery-v1'/row['path'];assert sha(q)==row['sha256'],str(q)
files=[p for p in H.rglob('*') if p.is_file() and p.name!='artifact-manifest.json' and '__pycache__' not in p.parts]
manifest=H/'artifact-manifest.json';manifest.write_text(json.dumps({'schema':1,'files':[rec(p) for p in sorted(files)]},indent=2)+'\n')
receipt={
 'schema':1,'task':'PAR-01 independent recovery review','owner':'editor_acceptance',
 'started_at_utc':'2026-09-13T20:52:58Z','ended_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'status':'verified CPU/source review of V1 bootstrap plus V2 watchdog; root launch admission pending',
 'acceptance':'PASS for the combined packet; frozen V1 watchdog alone is rejected',
 'scope':'One CPU affinity [0]; source and synthetic metadata/process tests only. No framework imports, model/weight reads, provider calls, credentials, network, allocations, GPU, final content, SSH, state or lease mutations. Original submitted payload and frozen recovery inputs unchanged.',
 'artifact_manifest':rec(manifest),
 'inputs':v['pins'],
 'runtime_source_pins':[rec(W/'r2-cloud-bootstrap-recovery-v1'/n) for n in ['cloud_entry.py','entry_observer.py','root-bound-entry.sh']]+[rec(W/'r2-cloud-bootstrap-recovery-v2'/n) for n in ['deadline_watchdog.py','root_arm_watchdog.py']],
 'verified':{
  'v1_artifact_pins':40,'core_source_files_byte_identical_to_submitted_payload':11,
  'core_source_id':'8ec908a41904888af647de2ee8b40ba8f73837777509d91159051b25fc72dd3e',
  'additional_byte_identical_files':['cloud_launch.py','cloud_train.py','runtime_setup.py','artifact_upload.py','requirements.txt','package-pins.json','trainer-source-manifest.json'],
  'training_contract':'Fresh Midtrain control, fixed full250 decision at 1000 horizon, micro2 accumulation8. Existing longest-row backward profile/runtime setup/uploader source unchanged; no optimizer/profile behavior change in the recovery patch.',
  'installed_resolver':'Three methods match actual installed source AST and source hash. Independent CLI argv-to-resolver tests validate status and terminate using ID alone or name plus explicit cloud.',
  'v2_guard':'monitor/validate/main AST equal to accepted delayed-submission guard16d3e01b. Only selector argv differs; root-arm pins exact V2 sibling guard and validates <=armed+7200 before exec.',
  'observer':'Provider stdout now reports shell, entry, setup admission/terminal and bounded training heartbeat. Only fixed error reasons, bounded filtered setup phrases and numeric telemetry are emitted; private upload logs are not tailed. Tests use synthetic markers only.',
  'source_stability':'All original V1 artifact hashes and both V2 guard pins reverified immediately before review freeze.'
 },
 'confirmed_v1_blocker':{
  'watchdog_sha256':'a55f0fc7b205d2056358d48ee868997776c8b789c4fd8e9be64bec15c4f8a89a',
  'failure':'No-ID failed status calls trigger termination at simulated140s, before a job appears at300s. This drops the accepted delayed-submission fix.',
  'correction':'Use V2 guard0e1a2d8a and root-arm056926f4 with V1 cloud entry/observer/shell. V2 retains unique-name monitoring until deadline before any exact job ID; once observed, existing identity and failure behavior remains.',
  'evidence':rec(H/'delayed-submission-evidence.json')
 },
 'tests':{
  'total_passed':35,'total_failed':0,'replayed_existing':28,'independent':7,
  'replay_seconds':2.543,'independent_seconds':0.007,
  'replay_log':rec(H/'replay-tests.log'),'independent_log':rec(H/'independent-tests.log'),
  'explicit_exclusion':'One unchanged real_torch_backward_profile test was not rerun because this review excludes model/framework imports; prior accepted CPU profile evidence is retained, not expanded.',
  'fixture_correction':'The first independent run exposed two test-fixture errors: a positional SDK stub and an omitted min(4200, remaining) expectation. Only owned tests were corrected; the initial log is preserved.',
  'commands':[
   f"cd {H} && CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /home/m0hawk/Documents/Sepalith/.venv-sft/bin/python -B verify_and_replay.py",
   f"cd {H} && CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /home/m0hawk/Documents/Sepalith/.venv-sft/bin/python -B test_independent.py"
  ]
 },
 'concrete_root_adoption_criteria':[
  'Independent worker confirms failed allocation release and billing; root retains the same total USD28 reservation. This review makes no provider or billing claim.',
  'Use a fresh UUID/job name/private artifact prefix and admitted root payload. Preserve selected TRAIN/draw/DEV/renderer/tokenizer/model identity; do not reuse failed admission or armed-receipt identity.',
  'Keep absolute_deadline_utc exactly2026-09-13T22:25:52Z from the original reservation. Fresh arming does not reset the window. Provider timeout remains6900 and watchdog <=armed+7200; inner setup/training/upload derive from the earlier absolute deadline.',
  'Stage V1 cloud_entry.py, entry_observer.py and root-bound-entry.sh, plus V2 deadline_watchdog.py and root_arm_watchdog.py. Keep guard and arm as siblings beside the two imported V1 Python files, or explicitly set PYTHONPATH to pinned V1 directory.',
  'Rehash the complete new payload and binding, arm the corrected watchdog before submission, bind its actual armed receipt hash, then submit with the existing fixed-node/provider timeout route.',
  'Observe entry-shell/entry-start/phase-terminal events. Bootstrap, runtime package integrity, actual allocated hardware fit, first optimizer step, private artifact upload/readback and independent provider release remain live acceptance checks.'
 ],
 'root_arming_command_template':'python3 -B ROOT_STAGED_PAYLOAD/root_arm_watchdog.py ROOT_FRESH_BINDING.json --output ROOT_FRESH_GUARD_DIRECTORY',
 'provider_entrypoint':'bash root-bound-entry.sh',
 'limits':[
  'The original remote startup failure cause remains unknown. Improved bootstrap diagnostics do not establish that pip/uv/compiler/runtime setup is repaired.',
  'This is source and CPU lifecycle evidence, not cloud capacity, A10 fit, successful training, private persistence or resource-release proof.',
  'The existing watchdog performs bounded20s status/terminate calls and up to120s confirmation; provider terminal observation is not independently sufficient for billed resource release. Root release/billing lane remains authoritative.',
  'No whole transitive wheel/OS closure or newly allocated environment validation is claimed. Exact unchanged package/runtime gates remain in force.'
 ],
 'next':'Root may admit one bounded diagnostic retry only after independent release/billing pass and exact new payload/watchdog binding.',
 'lease_released':True,'active_owned_processes':[]
}
p=R/'PAR-R2-cloud-bootstrap-independent-review.json';p.write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps({'receipt':rec(p),'artifact_manifest':rec(manifest)},indent=2))
