from pathlib import Path
import hashlib,json,datetime,fcntl,socket,subprocess
campaign=Path('docs/campaign');root=campaign/'work/lead/daily-lan-smoke-a';now=datetime.datetime.now(datetime.timezone.utc).isoformat();sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
installed=json.loads((root/'notebook-installed-file-observation.json').read_text())
assert installed['different_files']==['package.json']
import zipfile
with zipfile.ZipFile(campaign/'work/daily-lan-launch-preparation-v2/candidate.vsix') as z:expected=json.loads(z.read('extension/package.json'))
actual=installed['actual']['package'];assert set(actual)-set(expected)=={'__metadata'};assert all(actual[k]==v for k,v in expected.items())
(root/'notebook-installed-file-check.json').write_text(json.dumps({'installed_extension':installed['installed_extension'],'file_count':len(installed['expected_file_hashes']),'exact_file_matches':len(installed['expected_file_hashes'])-1,'package_json_exception':'VS Code added only __metadata; every packaged key/value matches exactly. Runtime bundle and all other files match VSIX hashes.','observation_sha256':sha(root/'notebook-installed-file-observation.json'),'status':'installed_payload_verified'},indent=2)+'\n')
def identity(pid):
 try:
  p=Path('/proc')/str(pid);s=(p/'stat').read_text().rsplit(')',1)[1].split();return None if s[0] in ('Z','X','x') else {'pid':pid,'startTick':s[19],'uid':p.stat().st_uid}
 except FileNotFoundError:return None
runs=[];ids=[]
for phase in ('first','reconnect'):
 d=json.loads((root/(phase+'-accepted-smoke.json')).read_text());session=Path(d['session']);assert d['terminal']['failure'] is None and d['terminal']['ledgerRelease']['status']=='released'
 for i in d['owned_identities_absent']:assert identity(i['pid'])!=i;ids.append(i)
 assert 'offloaded 43/43 layers to GPU' in (session/'native.log').read_text()
 runs.append({'phase':phase,'session':str(session),'seconds':d['terminal']['seconds'],'instance':d['result']['instance_id'],'responses':len(d['result']['outputs']),'response_sha256':[x['raw_sha256'] for x in d['result']['outputs']],'terminal_sha256':sha(session/'terminal.json'),'gateway_log_sha256':sha(session/'gateway-events.jsonl'),'native_log_sha256':sha(session/'native.log'),'client_evidence_sha256':sha(root/(phase+'-accepted-smoke.json'))})
for port in (18403,18423):
 with socket.socket() as s:s.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1);s.bind(('127.0.0.1',port))
with open('/home/m0hawk/.local/state/sepalith/campaign-20260915/resource-locks/cuda0.lock','r+') as f:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
terminal=json.loads((root/'controller-terminal.json').read_text());assert terminal['failure'] is None and terminal['completed_sessions']==2
r={'task_id':'RUN-01','owner':'lead','observed_at':now,'status':'bounded_daily_lan_smoke_accepted_resources_released','decision':'Accept daily v2 launcher start/stop/reconnect mechanics and the duration-only gateway amendment for final package preparation. Release and final editor benchmark remain pending.','inputs':[{'path':str(p),'sha256':sha(p)} for p in [campaign/'receipts/RUN-01-daily-lan-launch-preparation-v2.json',campaign/'receipts/RUN-01-daily-lan-independent-review-v1.json',root/'notebook-install.json',root/'notebook-installed-file-check.json',root/'smoke-client-pins.json',root/'deployment-admission.json']],'runs':runs,'denominators':{'sessions':2,'distinct_uuid':2,'train_fixture_cases':1,'cold_responses':2,'warm_responses':2,'complete_canonical_eos_responses':4,'exact_prompt_id_checks':2,'bound_http_responses':10,'owned_identities_rechecked_absent':len(ids)},'resource_release':{'identities':ids,'desktop_ports_reusable':[18403,18423],'notebook_ports_reusable':[18403,19463],'global_cuda_lock_reacquired':True,'ledger_own_row_released_both':True},'budget':{'outer_reservation_seconds':300,'actual_controller_seconds':terminal['seconds'],'unused_seconds_unallocated':300-terminal['seconds'],'cloud_spend':0},'limitations':['TRAIN fixture, not final quality','HTTP timings exclude editor/provider work; no representative p95','Actual extension Remote CUDA handshake on these daily bytes remains for final editor check','Session duration option source/CPU verified; no multi-hour soak claimed','Installed dedicated profile only; normal user profile unchanged; no release promotion'],'next':'Final package preparation and post-freeze actual daily editor benchmark; preserve b4 rollback.'}
out=campaign/'receipts/RUN-01-daily-lan-smoke-a-lead-acceptance.json';out.write_text(json.dumps(r,indent=2)+'\n')
ledger=campaign/'RESOURCE-LEASES.md';rows=ledger.read_text().splitlines();rows=[f'| Notebook CPU / editor | None | RUN-01 daily smoke terminal; SSH clients ended and ports reusable | {now} | Root admission before next notebook job |' if line.startswith('| Notebook CPU / editor |') else line for line in rows];ledger.write_text('\n'.join(rows)+'\n')
subprocess.run(['python3','docs/campaign/campaign.py','update','RUN-01','--status','done','--owner','lead','--note','Daily LAN v2 live start/stop/reconnect accepted: two distinct sessions, four complete matching TRAIN responses, twelve owned identities absent, CUDA lock and ledger released. Installed VSIX payload verified; only VS Code package metadata added. Final daily editor benchmark and release remain pending.','--receipt','receipts/'+out.name],check=True,stdout=subprocess.DEVNULL)
print(out,terminal['seconds'],len(ids))
