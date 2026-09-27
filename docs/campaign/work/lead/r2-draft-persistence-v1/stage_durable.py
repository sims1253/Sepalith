"""Stage only hash-bound profile outputs for the existing private uploader."""
import argparse,hashlib,json,pathlib,shutil

def digest(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
 return h.hexdigest()

def stage(run,destination):
 run=pathlib.Path(run).resolve();destination=pathlib.Path(destination)
 if destination.exists():raise ValueError('Use a fresh staging directory')
 manifest_path=run/'durable/persistence-manifest.json';manifest=json.loads(manifest_path.read_text())
 terminal=json.loads((run/'profile-terminal.json').read_text())
 if terminal.get('status')!='profile_passed':raise ValueError('Profile has not passed')
 selected=[];seen=set()
 for row in manifest['files']:
  rel=pathlib.PurePosixPath(row['path'])
  if rel.is_absolute() or '..' in rel.parts or str(rel) in seen:raise ValueError('Invalid or duplicate artifact path')
  seen.add(str(rel))
  if rel.parts[0] not in {'durable','checkpoints','stages','logs'}:raise ValueError('Unapproved durable artifact root: '+rel.parts[0])
  src=run.joinpath(*rel.parts)
  if any(x.is_symlink() for x in [src,*src.parents] if x!=run.parent):raise ValueError('Artifact symlink is forbidden')
  if not src.is_relative_to(run) or src.stat().st_size!=row['bytes'] or digest(src)!=row['sha256']:raise ValueError('Artifact identity differs')
  selected.append((src,rel))
 destination.mkdir(parents=True)
 for src,rel in selected:
  dst=destination.joinpath(*rel.parts);dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dst)
  if digest(src)!=digest(dst):raise ValueError('Staging readback differs')
 for name in ['profile-terminal.json','durable/persistence-manifest.json']:
  dst=destination/name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(run/name,dst)
 return {'status':'durable_staged_not_uploaded','files':len(selected)+2,'source':str(run),'destination':str(destination)}
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('run');a.add_argument('destination');v=a.parse_args();print(json.dumps(stage(v.run,v.destination)))
