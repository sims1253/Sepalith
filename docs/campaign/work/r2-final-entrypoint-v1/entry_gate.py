"""Shared additive gate: actual UTC and explicit frozen weight/harness first."""
import datetime,json,os,sys
from pathlib import Path
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
NATIVE=HERE.parent/'r2-final-evaluator-v1'
sys.path.insert(0,str(NATIVE))
import final_row_gate as gate
import final_binding as binding
Q8='d269a9fb85cd19efa05c6bf0dc11ccaa0f931fc50b826d893d58ae65043e02db'

def release(freeze_path,harness_sha256):
    # The freeze receipt is control metadata. No other input path is inspected.
    freeze=json.loads(Path(freeze_path).read_text())
    gate._validate_freeze(freeze,expected_weights_sha256=Q8,
        expected_harness_sha256=harness_sha256,now=datetime.datetime.now(datetime.timezone.utc))
    return freeze

def source_graph(path,expected,freeze,field):
    if freeze.get(field)!=expected:raise ValueError('source closure not explicitly frozen:'+field)
    graph=json.loads(Path(path).read_text())
    binding.verify_source_closure(graph,expected)
    return graph

def immutable_json(path,value):
    path=gate._safe_metadata_path(Path(path))
    if not path.parent.is_dir():raise ValueError('existing output parent required')
    raw=(json.dumps(value,sort_keys=True,ensure_ascii=False,indent=2)+'\n').encode()
    with path.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())

def source_epoch(graph):
    return {k:(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
        for k,v in graph['files'].items() for s in [Path(v['path']).stat()]}

def check_epoch(graph,before):
    if source_epoch(graph)!=before:raise ValueError('source changed during construction/evaluation')
