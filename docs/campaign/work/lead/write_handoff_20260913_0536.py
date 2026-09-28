from pathlib import Path
import datetime as dt
import hashlib
import json
import os
import subprocess

P=Path('/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb/docs/campaign')
f=P/'receipts/LEAD-HANDOFF.json'
old=json.loads(f.read_text())
now=dt.datetime.now(dt.timezone.utc)
backup=P/'work/lead'/('LEAD-HANDOFF-before-'+now.strftime('%Y%m%dT%H%M%SZ')+'.json')
backup.write_bytes(f.read_bytes())
state=json.loads((P/'state.json').read_text())
queue=P/'work/serving-hillclimb/reset-queue-v3-20260913'
qstatus=json.loads((queue/'status.json').read_text())
assert Path('/proc/3859998/cmdline').exists()
new_receipts=[
 'SFT-06-postmortem-lead-review.json',
 'RUN-09-theta0-semantic-lead-review.json',
 'RUN-09-theta0-native-quant-lead-decision.json',
 'RUN-09-theta0-q8-notebook-vulkan-lead-review.json',
 'RUN-09-theta0-q8-notebook-cpu-dev-partial-lead-review.json',
 'RUN-04-prewarm-v3-native-pilot-lead-review.json',
 'RUN-04-prewarm-recovery-a-lead-review.json',
 'RUN-04-cache-ram-ab-a-lead-decision.json',
 'RUN-04-theta0-editor-managed-a-lead-review.json',
 'RUN-04-theta0-editor-managed-a-resource-release.json',
 'RUN-05-theta0-context-trace-lead-review.json',
 'RUN-05-theta0-cuda-context-b-resource-release.json',
]
accepted=[]
for name in new_receipts:
 q=P/'receipts'/name
 assert q.exists(),q
 accepted.append({'path':'receipts/'+name,'sha256':hashlib.sha256(q.read_bytes()).hexdigest()})
value={k:old[k] for k in ['goal','workspaces','hard_cutoffs','source_snapshots']}
value.update({
 'observed_at':now.isoformat(),
 'state_revision':state['revision'],
 'previous_handoff_preserved':str(backup),
 'authority':'Lead owns launch, resource leases, checkpoint selection and release. Existing user authorization persists; no user information currently blocks local work.',
 'remaining_calendar_hours':{k:(dt.datetime.fromisoformat(v.replace('Z','+00:00'))-now).total_seconds()/3600 for k,v in old['hard_cutoffs'].items()},
 'source_heads':{'planning':'d7f2aa94981d57cd51f5dfd6573e4407e083ff94','canonical':'73d0bfef01adfe7288bd56d44f4a32d3d8f9e167','active_experiment_owner':'a7345e35219ceb624957115c39b869101026a817','dirty_sources':'Existing dirty files remain intact; launch manifests and preserved SFT/RL/extension snapshots bind required changes. No model or production source changed during current runtime review.'},
 'accepted_tasks':[k for k,v in state['tasks'].items() if v.get('status')=='done'],
 'accepted_artifacts':list(dict.fromkeys(old.get('accepted_artifacts',[])+new_receipts)),
 'latest_reviewed_receipts':accepted,
 'selection':{
  'status':'provisional; no final weight/harness freeze or release',
  'model':'SFT-primary-step1000-theta0',
  'HF_path':'/home/m0hawk/.local/state/sepalith/campaign-20260915/models/SFT-primary-step1000-theta0',
  'HF_weight_sha256':'499b7fdadfa701c04a4fba8f1717eb3ce02acc453237718562f002486d09840d',
  'Q8_path':'models/SFT-primary-step1000-quant-candidates-c/model-Q8_0.gguf',
  'Q8_sha256':'22401b9f6203cfcb8daa0f5a2919ba74e5e862889050cfb3059ceaf923fb4559',
  'renderer':'PRM03; current renderer-parity-v3 VSIX',
  'VSIX_sha256':'a56d2e442ae7abf266ef5f508d25443d00acecb9a78f92787b7ac332e3e2836d',
  'notebook_profile':'Vulkan b10453, context4096, output cap192, t6/tb6, b256/ub256, one slot; provisional final runtime',
  'fallback':'Protected b4 Q8 and matched legacy renderer; RUN-03-b4-host-baseline.json is accepted evidence.',
  'cut_decisions':'Later SFT/RL/corrective stages did not materially beat theta0; no unchanged training restart. Q6 timing cut after quality loss; cache-ram0 cut after 3 matched pairs showed no consistent gain.'},
 'active_resource_ownership':{
  'local_cuda':{'owner':None,'status':'RUN-05 context-b terminal clean; all10 owned PIDs absent, port18403 free, GPU0%, Windows occupancy3248MiB at05:29:53UTC. No CUDA workload admitted.','receipt':'RUN-05-theta0-cuda-context-b-resource-release.json'},
  'notebook':{'owner':None,'status':'CPU DEV terminal partial; editor terminal39.470327s; all20 editor PIDs gone. SSH recheck05:26UTC found CPU/guard PIDs gone and ports18401/2/3/13 free.','receipt':'RUN-04-theta0-editor-managed-a-resource-release.json'},
  'external':{'owner':'lead','pid':3859998,'exec_session':63075,'status':qstatus,'not_before':'2026-09-13T07:02:00Z','Berlin_not_before':'2026-09-13T09:02:00+02:00','hard_stop':'2026-09-13T08:00:00Z','job':'opus16-ordinary-q8-prefill-after-reset','max_calls':1,'max_call_seconds':1800,'effort':'high','packet':'work/serving-hillclimb/opus16-ordinary-q8-prefill-after-reset.txt','packet_sha256':'7112919db1ea9d5d7dd5ab23f25eb82ba305be3df337fa179bf31d433b4a5c43','receipt':'RUN-09-opus-reset-queue-v3.json','stop_rule':'First failure stops queue. No automatic patch application, native launch or promotion.'},
  'local_cpu_unmanaged':{'pid':3759393,'owner':'Existing user VSCode extensionHost','port':18099,'model':'abl_dropout-Q8_0.gguf','GPU_layers':0,'action':'Left intact; CPU contention must be disclosed.'},
  'paid_cloud':{'active_jobs':0,'spend_USD':0,'spend_EUR':0}},
 'workers':{'corrected_train_review':'Completed selected-theta0 editor preparation; root reviewed, launched and accepted limited evidence.','rl_contexts':'Completed CPU DEV partial independent review; root replayed and accepted partial diagnostic only.','rl_entry':'Completed context trace review v2; root caught and corrected v1 units/denominator error, verified v2. All workers idle.'},
 'latest_findings':[
  'Local CUDA DEV75: Q8 and F16 each26/43 exact edits,25/32 correct no-ops,5 false suggestions. Q6 has24 edits24 no-ops; Q6 timing remains cut.',
  'Notebook Vulkan Q8 DEV75:26/43 exact edits,23/32 correct no-ops,7 false suggestions. CPU bounded run:34 complete,1 partial,40 unattempted; observed only8 roxygen and26 no-op. No full CPU quality ranking or backend promotion.',
  'CPU observed native prefill50.34tokens/s for2619tokens; not editor latency. Charge988.593seconds conservatively. No additional full CPU DEV run allocated.',
  'Selected-theta0 managed notebook editor:16 assertions passed,3 pending; one actual inline commit changed an unsaved synthetic buffer to result <- value + 1, and exact committed text passes R parse. Four production provider response logs1169-1481ms; these are not first-visible latency. All owned processes stopped.',
  'Native prewarm short TRAIN pair improved foreground2.178s to0.697s with identical output. Long1903-token fixture missed5s. Recovery short request completed3.538s after long cancellation. Prewarm remains unpromoted pending real editor and repeat evidence.',
  'Cache-ram0 versus default8192,3 matched pairs: no consistent gain despite removing about1.1s logged archive maintenance. Keep default cache setting; cut candidate0.',
  'CUDA context-a inference completed but shutdown exits1/-6/-11 failed. Process fixture reproduced duplicate TERM callbacks. New context-b supervisor uses wrapper-only TERM and timeout --foreground; all3 server/client exits0 and no survivors.',
  'Root HF encode/decode verified all54 A+B raw responses and exact prompt/output/plan parity across2K/4K/8K. Corrected run has24 fresh plus3 repeat responses,62-133ms fresh request times on148-300-token prompts. This establishes short allocation behavior only.',
  'Corrective SFT06 is closed at100 after regressions. Unused1775.576667825seconds released unallocated. No new target-only-loss or lower-LR proposal admitted.',
  'Final constructor/production integration/quality factory passed synthetic CPU preparation gates. No sealed final content was read; final evaluation needs both Monday10UTC and explicit root weight/harness freeze.'
 ],
 'unresolved':[
  'RUN-04 visible ghost and stale/cancelled nonpublication, multiline acceptance, and first-visible editor latency remain open. Unchanged document content alone does not prove nonpublication.',
  'RUN-05 meaningful long2K/4K source/event traces and8K stress coverage remain open. Native short8K allocation does not support8K-token prompt or quality claims. Keep4K primary default.',
  'Prewarm is not production-promoted; first-party Opus ordinary Q8 prefill proposal must be reviewed and tested before any build/runtime admission.',
  'DSpark Opus12 is CPU-reviewed only; mandatory safety amendment not yet staged/native-tested. Q6 compiled artifacts preserved but timing cut.',
  'DAT-08/REL-02 complete transitive source closure and actual loaded tensor binding remain required. Final model/quant/draft/runtime checks must use selected frozen weights.',
  'REL-00/REL-01 training and optional branch closure, Monday freeze and final release decisions remain root-owned. Do not reopen cut training just to use idle hardware or unspent ceiling.',
  'Goal tool still reports blocked and only supports complete/blocked mutations. No false completion recorded; authorized work continues.',
  'Historical filename-only scan across NATIVE found0 matches and opened no final content. Continue exact-path artifact access; do not enumerate sealed-final.'
 ],
 'budget':{
  'campaign_cloud_spend_USD':0,'campaign_cloud_spend_EUR':0,'Anyscale_verified_balance_USD':95.433820846,'Anyscale_expiry':'2126-08-05','Anyscale_infrastructure_credit_coverage':'verified; no billing question pending','Anyscale_all_in_ceiling_USD':60,'Anyscale_remaining_admission':'capacity, persistence and termination controls before any paid allocation',
  'Azure_all_in_ceiling_EUR':100,'Azure_capacity':'12regions/216entries audited; GPU quotas0, branch closed unless new capacity evidence',
  'Kaggle_GPU_hours_verified':30,'Kaggle_TPU_hours_verified':20,'Kaggle_observed_at':'2026-09-12T05:00:37UTC','Kaggle_jobs':0,
  'RL_all_in_seconds_ceiling':93600,'RL_charged_seconds':16241.972066079034,'SFT06_corrective_seconds_charged':1824.423332175007,'RL_unreserved_remaining_seconds':75533.60460174596,'SFT06_unused_seconds_unallocated':1775.576667824993,'draft_training_seconds_allocated':0,
  'native_quant_DEV_seconds_charged':141.900619,'notebook_Vulkan_DEV_seconds_charged':813.8758590460056,'notebook_CPU_partial_seconds_charged':988.593,'notebook_editor_seconds_charged':39.470327340008225,'notebook_prewarm_pilot_seconds_charged':32.163488250007504,'notebook_recovery_seconds_charged':13.46434304700233,'notebook_cache_AB_seconds_charged':83.80379008200543,'local_CUDA_context_a_plus_b_seconds_charged':19.845201908028685,
  'Q6_native_timing_seconds_cut':1200,'cache_AB_unused_seconds_unallocated':516.1962099179946,'context_b_unused_seconds_unallocated':293.2791694140178,
  'disk_free_bytes_observed':{'native':164824702976,'E':1896190210048,'NAS':2922172121088},
  'allocation_rule':'All unused ceilings are unallocated; elapsed-time ceilings are not permission for new optional work.'},
 'pending_user_info':[],
 'next_concrete_actions':[
  'RUN-04: finish actual ghost/cancellation nonpublication, multiline commit and first-visible latency checks on selected Q8 notebook route; use synthetic or training evidence only.',
  'RUN-05: prepare bounded long2K/4K and8K stress traces with corrected timeout supervisor, explicit overflow rejection and fixed output cap. Lease notebook or CUDA lane before launch. Keep4K default absent evidence.',
  'At07:02UTC, monitor the existing Opus16 queue and review final result/artifact. Stop on first failed call; no automatic application. Ordinary Q8 prefill is the only queued source candidate.',
  'Before Monday freeze, finish REL-02 source/model binding and preserve finalist plus b4 rollback. No sealed content before2026-09-14T10:00Z AND explicit root freeze receipt.',
  'At main cutoff Sep13 22UTC, close training/optional decisions explicitly. At Monday freeze, select by DEV evidence and then run sealed final evaluation and final runtime/quant/draft checks.'
 ],
 'board':'docs/campaign/board.html is the standalone readable snapshot; campaign CLI/state.json is authoritative. PostPlan is a snapshot.'
})
f.write_text(json.dumps(value,indent=2)+'\n')
ledger=P/'RESOURCE-LEASES.md';lines=ledger.read_text().splitlines()
for i,line in enumerate(lines):
 if line.startswith('| Local CPU |'):lines[i]=f'| Local CPU | lead | Runtime review complete; all bounded workers idle | {now.isoformat()} | User CPU server left intact |'
 if line.startswith('| External queue |'):lines[i]=f'| External queue | lead | Controller3859998 alive; heartbeat {qstatus["at"]}; no active Claude call | {now.isoformat()} | One Opus16 call07:02UTC/1800s; hardstop08UTC; no autoapply |'
ledger.write_text('\n'.join(lines)+'\n')
print(json.dumps({'handoff':str(f),'revision':state['revision'],'sha256':hashlib.sha256(f.read_bytes()).hexdigest(),'remaining_hours':value['remaining_calendar_hours']}))
