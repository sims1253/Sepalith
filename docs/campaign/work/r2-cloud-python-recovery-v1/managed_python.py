"""Validate pinned uv discovery without treating interpreter aliases as installs."""
import os
from pathlib import Path
PROBE="import json,sys,sysconfig; from pathlib import Path; print(json.dumps({'version':list(sys.version_info[:3]),'implementation':sys.implementation.name,'executable':str(Path(sys.executable).resolve()),'prefix':str(Path(sys.prefix).resolve()),'header':str((Path(sysconfig.get_path('include'))/'Python.h').resolve())}))"
def require(ok, reason):
 if not ok: raise ValueError(reason)
def owned(path, root):
 p=Path(path)
 require(p.is_absolute(),'managed Python path is not absolute')
 p=p.resolve(strict=True)
 require(p.is_relative_to(Path(root).resolve(strict=True)),'managed Python path outside owned root')
 return p
def discover_result(output,root):
 lines=[x.strip() for x in output.splitlines() if x.strip()]
 require(len(lines)==1,'managed Python discovery must return one path')
 p=owned(lines[0],root)
 require(p.is_file() and os.access(p,os.X_OK),'managed Python executable missing')
 return p
def validate_runtime(record,interpreter,root):
 require(record.get('version')==[3,10,19] and record.get('implementation')=='cpython','managed Python runtime version differs')
 require(owned(record['executable'],root)==Path(interpreter).resolve(strict=True),'managed Python runtime executable differs')
 require(owned(record['prefix'],root).is_dir(),'managed Python runtime prefix missing')
 require(owned(record['header'],root).is_file() and Path(record['header']).name=='Python.h','managed Python development headers missing')
 return True
def layout_diagnostics(root):
 root=Path(root);matches=list(root.glob('*/bin/python3.10'))
 # Only bounded counts and layout types; never expose environment, paths or file content.
 return {'legacy_glob_count':len(matches),'legacy_unique_resolved_count':len({str(p.resolve()) for p in matches}), 'top_directory_count':sum(p.is_dir() for p in root.iterdir()),'top_symlink_count':sum(p.is_symlink() for p in root.iterdir())}
