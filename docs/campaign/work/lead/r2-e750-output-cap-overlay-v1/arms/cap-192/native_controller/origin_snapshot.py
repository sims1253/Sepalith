"""Root-authorized CPU import observation only; never loads models or calls CUDA."""
import hashlib, importlib.util, json, marshal, os, pathlib, signal, subprocess, sys, time, types
HERE = pathlib.Path(__file__).resolve().parent
SITE = '/home/m0hawk/Documents/Sepalith/.venv-sft/lib/python3.10/site-packages'

def label(value):
    cls = value if isinstance(value, type) else type(value)
    return cls.__module__ + '.' + cls.__qualname__

def code_value(value):
    if isinstance(value, types.CodeType):
        return {'code': value.co_code.hex(), 'consts': code_value(value.co_consts),
                'names': value.co_names, 'varnames': value.co_varnames, 'freevars': value.co_freevars,
                'cellvars': value.co_cellvars, 'argcount': value.co_argcount,
                'posonly': value.co_posonlyargcount, 'kwonly': value.co_kwonlyargcount,
                'flags': value.co_flags, 'stacksize': value.co_stacksize,
                'filename': value.co_filename, 'name': value.co_name, 'firstlineno': value.co_firstlineno,
                'linetable': value.co_linetable.hex()}
    if isinstance(value, (tuple, list)): return [code_value(x) for x in value]
    if isinstance(value, frozenset): return {'frozenset': sorted((code_value(x) for x in value), key=lambda x:json.dumps(x,sort_keys=True))}
    if isinstance(value, bytes): return {'bytes':value.hex()}
    if isinstance(value, complex): return {'complex':repr(value)}
    if value is Ellipsis: return {'ellipsis':True}
    if isinstance(value, (str, int, float, bool, type(None))): return value
    raise TypeError('unsupported code constant '+type(value).__name__)

def code_record(function):
    code = function.__code__
    closure = []
    for cell in function.__closure__ or ():
        try: value = cell.cell_contents
        except ValueError: value = None
        closure.append(value if isinstance(value, (str, int, bool, type(None))) else {'type': label(value)})
    return {'filename':code.co_filename,'name':code.co_name,
            'code_sha256':hashlib.sha256(json.dumps(code_value(code),sort_keys=True,separators=(',',':')).encode()).hexdigest(),'closure':closure}

def snapshot():
    names = {}
    for name, module in sys.modules.items():
        names.setdefault(id(module), []).append(name)
    records=[]
    for name,module in sorted(list(sys.modules.items())):
        if module is None: continue
        v=vars(module);spec=v.get('__spec__');loader=v.get('__loader__')
        record={'name':name,'type':label(module),'file':v.get('__file__'),'cached':v.get('__cached__'),
                'spec_origin':getattr(spec,'origin',None),'loader':None if loader is None else label(loader),
                'same_object_names':sorted(names[id(module)])}
        if spec is not None and spec.submodule_search_locations is not None:
            record['namespace_paths']=[str(x) for x in spec.submodule_search_locations]
        if not record['file'] or not pathlib.Path(record['file']).is_file() or record['spec_origin'] in ('built-in','frozen'):
            record['functions']={k:code_record(x) for k,x in v.items() if isinstance(x,types.FunctionType)}
            record['module_attributes']={k:sorted(names[id(x)]) for k,x in v.items() if isinstance(x,types.ModuleType) and id(x) in names}
            record['wrapped_modules']={k:{'name':vars(x).get('__name__'),'file':vars(x).get('__file__')}
                 for k,x in v.items() if isinstance(x,types.ModuleType) and id(x) not in names}
            if loader is not None and isinstance(vars(loader).get('data'),str):
                text=vars(loader)['data'];record['generated_source_sha256']=hashlib.sha256(text.encode()).hexdigest()
                record['generated_source_bytes']=len(text.encode())
        records.append(record)
    return {'status':'observed_not_admitted','modules':records,'sys_path':sys.path,
            'interpreter':sys.executable,'version':sys.version,'flags':str(sys.flags),
            'builtin_module_names':list(sys.builtin_module_names),'final_access':False,'model_load':False}
