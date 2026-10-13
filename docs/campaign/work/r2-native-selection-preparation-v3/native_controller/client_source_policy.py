"""Trusted fresh interpreter/factory closure; no arbitrary-code sandbox claim."""
import os, sys
from pathlib import Path
import origin_snapshot

def executable_maps():
    files=set()
    for line in Path('/proc/self/maps').read_text().splitlines():
        fields=line.split(None,5)
        if 'x' not in fields[1]:continue
        name=fields[5] if len(fields)==6 else ''
        if name in ('[vdso]','[vsyscall]'):continue
        if not name.startswith('/') or name.endswith(' (deleted)'):raise ValueError('unreviewed executable mapping')
        files.add(str(Path(name).resolve()))
    return sorted(files)

def verify(graph,*,deep):
    p=graph['runtime_policy']
    if not sys.flags.isolated or not sys.flags.no_site:raise ValueError('isolated no-site process required')
    if sys.version!=p['python_version'] or str(Path(sys.executable).resolve())!=p['interpreter_realpath']:
        raise ValueError('interpreter changed')
    if sys.path!=p['sys_path']:raise ValueError('search roots changed')
    if any(Path(path).exists() for path in p['must_remain_absent']):raise ValueError('new bytecode appeared')
    expected=p['modules']
    for name,module in list(sys.modules.items()):
        if module is None:continue
        if name not in expected:raise ValueError('unexpected module: '+name)
        row=expected[name]
        if not row['bindings'] or any(k not in graph['files'] for k in row['bindings']):raise ValueError('factory binding missing: '+name)
        observed=row['observation'];v=vars(module)
        if v.get('__file__')!=observed['file'] or v.get('__cached__')!=observed['cached'] or origin_snapshot.label(module)!=observed['type']:
            raise ValueError('module origin drift: '+name)
    if executable_maps()!=p['executable_paths']:raise ValueError('client executable map changed')
    if deep:
        current={m['name']:m for m in origin_snapshot.snapshot()['modules']}
        for name,observation in current.items():
            if observation!=expected[name]['observation']:raise ValueError('module factory/alias drift: '+name)
    return {'runtime_modules':sum(m is not None for m in sys.modules.values()),'deep':deep}
