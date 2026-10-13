"""Bounded provider-stdout observations; never tail private upload logs."""
import datetime,json,math,re,time
from pathlib import Path
SETUP_PHASES=frozenset(('uv-bootstrap','python-bootstrap','venv-bootstrap','packages-bootstrap','packages-check','runtime-setup'))
def emit(event,**values):
 print(json.dumps({'event':event,'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),**values},allow_nan=False,sort_keys=True),flush=True)
def safe_reason(error):
 # Only authored control errors may reach stdout; arbitrary exception strings do not.
 if type(error) is ValueError and str(error) in SAFE_REASONS:return str(error)
 return type(error).__name__
SAFE_REASONS=set(['UTC offset missing', 'backward profile missing', 'comparison geometry differs', 'control must initialize fresh', 'credential fields forbidden', 'deadline layers differ', 'duplicate relocation', 'expired or invalid cloud window', 'first full250 stop differs', 'fresh Midtrain control differs', 'fresh UUID32 required', 'full250 task stop not reached', 'guarded phase failed', 'image compiler or GNU timeout missing', 'image or instance differs', 'insufficient guarded training time', 'managed Python3.10 missing or ambiguous', 'no phase budget remains', 'payload escaped root', 'payload hash differs', 'phase budgets differ', 'pinned uv executable missing', 'prelaunch deadline or Linux memory floor', 'private unique artifact prefix differs', 'recipe launch admission pending', 'recipe pin differs', 'relocation changed content identity', 'root payload admission is pending', 'root watchdog binding missing', 'symlink payload path', 'task horizon identity differs', 'trainer manifest pin differs', 'trainer source differs', 'unbound input relocation', 'unsafe payload path', 'watchdog limit expanded'])
def setup_tail(path,phase):
 if phase not in SETUP_PHASES:return []
 try:
  with Path(path).open('rb') as stream:
   stream.seek(0,2);length=stream.tell();stream.seek(max(0,length-8192));raw=stream.read(8192)
  lines=raw.decode('utf-8','replace').splitlines()
  if length>8192:lines=lines[1:]
 except OSError:return []
 out=[]
 for line in lines[-12:]:
  # Only fixed diagnostic phrases and validated public package requirements are
  # emitted. Unrecognized text, URLs, paths, credentials and tracebacks stay local.
  patterns=[r'(?i)(No module named) [\x27\x22]?([a-zA-Z][a-zA-Z0-9_.]{0,50})[\x27\x22]?$',r'(?i)(No matching distribution found for) ([a-zA-Z][a-zA-Z0-9_.-]{0,50}(?:==[0-9][0-9.a-z]{0,30})?)$']
  rendered=None
  for pattern in patterns:
   match=re.search(pattern,line)
   if match:rendered=' '.join(match.groups());break
  if rendered is None:
   for phrase in ('externally-managed-environment','Permission denied','No space left on device','No such file or directory','Connection timed out','Temporary failure in name resolution','CERTIFICATE_VERIFY_FAILED','command not found','Killed','requires a different Python'):
    if phrase in line:rendered=phrase+' [remaining text omitted]';break
  out.append(rendered if rendered is not None else '[unrecognized setup output omitted]')
 return out

def telemetry_snapshot(path):
 """Read at most64KiB; return only numeric completed-step/loss evidence."""
 try:
  with Path(path).open('rb') as stream:
   stream.seek(0,2);length=stream.tell();stream.seek(max(0,length-65536));raw=stream.read(65536)
  rows=raw.splitlines(keepends=True)
  if length>65536:rows=rows[1:]
 except OSError:return {'telemetry_state':'not_yet_present'}
 result={'telemetry_state':'no_complete_step_record'}
 for raw in rows:
  if not raw.endswith(b'\n'):continue
  try:row=json.loads(raw)
  except (ValueError,UnicodeError):continue
  if not isinstance(row,dict):continue
  step=row.get('step')
  if type(step) is not int or step<0:continue
  if row.get('event')=='optimizer_step':
   result['last_completed_step']=step;result['telemetry_state']='completed_step_observed'
   for key in ('seconds','remaining_seconds'):
    value=row.get(key)
    if type(value) in (int,float) and math.isfinite(value):result[key]=value
   resources=row.get('resources',{})
   if isinstance(resources,dict):
    result['trainer_resources']={k:v for k,v in resources.items() if k in ('process_id','process_peak_rss_bytes','cuda_allocated_bytes','cuda_reserved_bytes','cuda_attempt_peak_allocated_bytes','cuda_attempt_peak_reserved_bytes') and type(v) is int and v>=0}
  if row.get('event')=='trainer_metrics' and isinstance(row.get('metrics'),dict):
   metrics={k:v for k,v in row['metrics'].items() if k in ('loss','grad_norm','learning_rate') and type(v) in (int,float) and math.isfinite(v)}
   if metrics:result.update(last_metrics_step=step,metrics=metrics)
 return result

def rss_bytes(identity):
 if not identity:return None
 try:
  raw=Path(f'/proc/{identity["pid"]}/stat').read_text();parts=raw[raw.rfind(')')+2:].split()
  if int(parts[19])!=identity['start_tick']:return None
  for line in Path(f'/proc/{identity["pid"]}/status').read_text().splitlines():
   if line.startswith('VmRSS:'):return int(line.split()[1])*1024
 except (OSError,ValueError,IndexError):pass
 return None
