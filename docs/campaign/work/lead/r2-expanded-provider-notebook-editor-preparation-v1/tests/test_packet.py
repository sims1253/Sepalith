import hashlib,json,pathlib,re
ROOT=pathlib.Path(__file__).resolve().parents[1]
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def test_identity_and_defaults():
 assert sha(ROOT/'remote/candidate-extension/dist/extension.js')=='336d03ef0c61fdc510c3843a19a729562f43f6d340f3d9925d994a9d6abc59f4'
 assert json.loads((ROOT/'remote/candidate-extension/package.json').read_text())['contributes']['configuration']['properties']['sepalith.expandedProvider']['default'] is False
 assert "sepalith.expandedProvider':true" in (ROOT/'remote/run_editor_mechanics.mjs').read_text()
def test_private_launch_and_bounds():
 s=(ROOT/'remote/run_editor_mechanics.mjs').read_text()
 for x in ['--user-data-dir','--extensions-dir','--extensionDevelopmentPath=','fresh run-root required','maximum_seconds!==240','requestTimeoutMs\':5000']:
  assert x in s,x
 assert "taskset" not in s # owned by outer root command
 assert 'install-extension' not in s
 assert "process.kill(-c.pid" in s and "SIGKILL" in s
def test_train_fixtures():
 m=json.loads((ROOT/'remote/fixtures/fixtures.json').read_text()); assert len(m['rows'])==2
 for r in m['rows']:
  p=ROOT/'remote/fixtures'/r['relative_file'];assert sha(p)==r['preedit_sha256'];assert r['source_role']=='authorized TRAIN pre-edit only'
def test_mock_scenarios_and_no_quality_claim():
 s=(ROOT/'remote/mock-llama-server.mjs').read_text();h=(ROOT/'remote/editor-harness/extension.js').read_text()
 for x in ['edit','noop','malformed','delayed','client_close']:assert x in s+h
 for x in ['application geometry exact','document-version invalidation','cursor invalidation','total deadline leaves document unchanged','quality_or_latency_evidence:false','namespace identity invalidation suppresses stale edit']:assert x in h
