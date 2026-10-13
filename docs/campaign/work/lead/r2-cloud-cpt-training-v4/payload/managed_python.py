"""Pinned uv interpreter discovery without alias-count ambiguity."""
import json,os,subprocess
from pathlib import Path
PROBE="import json,sys,sysconfig;from pathlib import Path;print(json.dumps({'version':list(sys.version_info[:3]),'implementation':sys.implementation.name,'executable':str(Path(sys.executable).resolve()),'prefix':str(Path(sys.prefix).resolve()),'header':str((Path(sysconfig.get_path('include'))/'Python.h').resolve())}))"
def owned(path,root):
 p=Path(path).resolve(strict=True)
 if not p.is_relative_to(Path(root).resolve(strict=True)):raise ValueError('managed Python outside owned root')
 return p
def discover(uv,root,env,deadline,guard,cwd,log):
 guard([str(uv),'python','find','--managed-python','--no-python-downloads','--no-project','--resolve-links','3.10.19'],env,cwd,deadline,log)
 lines=[x.strip() for x in Path(log).read_text().splitlines() if x.strip()]
 if len(lines)!=1:raise ValueError('managed Python discovery differs')
 p=owned(lines[0],root)
 if not p.is_file() or not os.access(p,os.X_OK):raise ValueError('managed Python executable absent')
 return p
def validate(record,interpreter,root):
 if record.get('version')!=[3,10,19] or record.get('implementation')!='cpython':raise ValueError('managed Python runtime differs')
 if owned(record['executable'],root)!=Path(interpreter).resolve(strict=True):raise ValueError('managed Python executable differs')
 if not owned(record['header'],root).is_file():raise ValueError('managed Python headers absent')
