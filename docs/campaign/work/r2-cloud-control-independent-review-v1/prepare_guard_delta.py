import difflib,hashlib
from pathlib import Path
H=Path(__file__).resolve().parent;P=H.parent/'r2-cloud-admission-v1/deadline_watchdog.py'
old=P.read_text();assert hashlib.sha256(P.read_bytes()).hexdigest()=='3cb42137fef07301a8fcc96191d41a3efdc7bba4283d0c1faf3a9c9b891a31a5'
needle='  elif now-start>=120:\n   errors+=1'
assert old.count(needle)==1
new=old.replace(needle,"  elif last_id is None:\n   # Submission/upload may still be running. Keep the unique-name guard\n   # alive until the absolute deadline; do not abandon a job created later.\n   record({'event':'pending_submission','utc_epoch':clock(),'name':name,\n           'status':'no_exact_job_id_observed','absolute_deadline_epoch':deadline})\n  elif now-start>=120:\n   errors+=1")
q=H/'deadline_watchdog_delayed_submission.py';q.write_text(new)
(H/'watchdog-delayed-submission.patch').write_text(''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile='a/deadline_watchdog.py',tofile='b/deadline_watchdog.py')))
print({'candidate_sha256':hashlib.sha256(q.read_bytes()).hexdigest()})
