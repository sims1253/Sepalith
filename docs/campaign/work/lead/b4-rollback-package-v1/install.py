"""Install the pinned b4 extension into its dedicated recovery profile."""
import hashlib,json,subprocess,datetime
from pathlib import Path
HERE=Path(__file__).resolve().parent

def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()

def main():
 spec=json.loads((HERE/'rollback-spec.json').read_text());settings=(HERE/'settings.json').read_bytes()
 for row in [spec['model'],spec['matched_vsix'],*spec['runtime_files']]:
  p=Path(row['path'])
  if p.stat().st_size!=row['bytes'] or digest(p)!=row['sha256']:raise RuntimeError('rollback_artifact_mismatch:'+str(p))
 root=Path(spec['dedicated_user_data']);target=root/'User/settings.json'
 if target.exists() and target.read_bytes()!=settings:raise RuntimeError('existing_rollback_settings_differ')
 target.parent.mkdir(parents=True,exist_ok=True)
 if not target.exists():
  with target.open('xb') as f:f.write(settings)
 result=subprocess.run(spec['install_extension_argv'],capture_output=True,text=True,timeout=90)
 receipt={'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'exit':result.returncode,'stdout':result.stdout,'stderr':result.stderr,'settings_sha256':digest(target),'profile':str(root),'all_model_runtime_vsix_hashes_verified':True,'server_started':False}
 (HERE/'installation-result.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt));return result.returncode
if __name__=='__main__':raise SystemExit(main())
