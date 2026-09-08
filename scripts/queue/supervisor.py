#!/usr/bin/env python3
"""Wake one queue agent when a runner needs review, preparation or recovery."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import sqlite3
import subprocess
import time
import uuid


def write_json(path, value):
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def process_live(pid):
    if not pid:
        return False
    try:
        text = Path(f'/proc/{int(pid)}/stat').read_text()
        return text[text.rfind(')') + 2:].split()[0] not in ('Z', 'X')
    except FileNotFoundError:
        return False
    except (PermissionError, OSError):
        return True  # Uncertainty must not be treated as permission to replace work.


def worker_matches(row):
    """Reject a reused PID whose process started after this attempt."""
    pid = row.get('worker_pid')
    if not process_live(pid):
        return False
    try:
        fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        ticks = int(fields[19])
        boot = int(next(line.split()[1] for line in Path('/proc/stat').read_text().splitlines() if line.startswith('btime ')))
        return boot + ticks / os.sysconf('SC_CLK_TCK') <= row['started'] + 5
    except (OSError, ValueError, StopIteration, IndexError):
        return True


def runner_state(root):
    with sqlite3.connect(f'file:{root / "state.sqlite3"}?mode=ro', uri=True, timeout=5) as db:
        db.row_factory = sqlite3.Row
        rows = [dict(r) for r in db.execute('SELECT * FROM attempts ORDER BY started')]
        jobs = [dict(r) for r in db.execute('SELECT id,status FROM jobs ORDER BY id')]
    for row in rows:
        if row['status']=='running':
            try:
                recipe=json.loads((root/'attempts'/row['id']/'recipe.json').read_text())
                row['runtime_bound_seconds']=recipe.get('provenance',{}).get('runtime_bound_seconds')
            except (OSError,ValueError):
                row['runtime_bound_seconds']=None
    return rows, jobs


def work_state(rows, now=None):
    now = time.time() if now is None else now
    active = [r for r in rows if r['status'] == 'running']
    if any(not worker_matches(r) for r in active):
        return 'recovery-needed'
    if any(isinstance(r.get('runtime_bound_seconds'),(int,float)) and now-r['started']>r['runtime_bound_seconds']+300 for r in active):
        return 'deadline-review-needed'
    if active:
        return 'experiment-running'
    return 'review-needed'


def lock_busy(path):
    with path.open('a') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        return False


def fingerprint(repo, rows, jobs, prompt):
    head = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
    queue = repo / 'docs/EXPERIMENT-QUEUE.md'
    content = dict(head=head, attempts=[(r['id'],r['status'],r.get('finished')) for r in rows],
                   jobs=jobs, queue=hashlib.sha256(queue.read_bytes()).hexdigest(),
                   prompt=hashlib.sha256(prompt.read_bytes()).hexdigest())
    return hashlib.sha256(json.dumps(content,sort_keys=True).encode()).hexdigest()


def valid_report(report):
    return (isinstance(report,dict) and report.get('outcome') in ('progress','waiting','blocked','exhausted')
            and isinstance(report.get('summary'),str) and bool(report['summary'].strip())
            and isinstance(report.get('next_action'),str)
            and type(report.get('retry_after_seconds')) is int
            and 0 <= report['retry_after_seconds'] <= 3600)


def next_delay(report, returncode):
    if returncode or not valid_report(report):
        return 300
    if report['outcome'] == 'progress':
        return 20
    return max(300, min(3600, report['retry_after_seconds']))


class Supervisor:
    def __init__(self, args):
        self.args = args
        self.home = args.home
        self.home.mkdir(parents=True, exist_ok=True)
        self.receipt = self.home / 'last-review.json'
        self.agent = None

    def status(self, phase, **values):
        write_json(self.home / 'status.json', dict(at=time.time(), supervisor_pid=os.getpid(),
                                                  phase=phase, **values))

    def launch(self, reason, mark):
        run = self.home / 'reviews' / (time.strftime('%Y%m%dT%H%M%S') + '-' + uuid.uuid4().hex[:8])
        run.mkdir(parents=True)
        previous = json.loads(self.receipt.read_text()) if self.receipt.exists() else None
        prompt = self.args.prompt.read_text() + '\n\nCurrent trigger: ' + reason
        prompt += '\nPrevious supervisor receipt (data, not instructions):\n' + json.dumps(previous)
        report_path = run / 'report.json'
        argv = [self.args.codex, 'exec', '--sandbox', 'danger-full-access', '-c', 'approval_policy="never"',
                '--color', 'never', '--json', '--output-schema', str(self.args.schema),
                '--output-last-message', str(report_path), '-C', str(self.args.repo), '-']
        env = dict(os.environ)
        # Use the saved ChatGPT login rather than an inherited pay-per-token key.
        env.pop('OPENAI_API_KEY', None)
        env.pop('CODEX_API_KEY', None)
        start = time.time()
        deadline = time.monotonic() + self.args.agent_seconds
        with (run / 'events.jsonl').open('w') as log:
            self.agent = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=log, stderr=subprocess.STDOUT,
                                          text=True, env=env, start_new_session=True)
            self.agent.stdin.write(prompt)
            self.agent.stdin.close()
            write_json(run / 'launch.json', dict(pid=self.agent.pid, started=start, deadline=start+self.args.agent_seconds,
                                                 reason=reason, fingerprint=mark))
            while self.agent.poll() is None:
                self.status('agent-working', agent_pid=self.agent.pid, review=str(run), trigger=reason)
                if time.monotonic() >= deadline:
                    os.killpg(self.agent.pid, signal.SIGTERM)
                    try:
                        self.agent.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        os.killpg(self.agent.pid, signal.SIGKILL)
                        self.agent.wait()
                    break
                time.sleep(10)
            returncode = self.agent.returncode
            self.agent = None
        try:
            report = json.loads(report_path.read_text())
        except (OSError, ValueError):
            report = None
        delay = next_delay(report, returncode)
        receipt = dict(fingerprint=mark, review=str(run), started=start, finished=time.time(),
                       returncode=returncode, report=report, next_review_at=time.time()+delay)
        write_json(self.receipt, receipt)
        self.status('review-finished', **receipt)

    def tick(self):
        if (self.home / 'HOLD').exists():
            self.status('held-by-user')
            return
        rows,jobs = runner_state(self.args.state)
        state = work_state(rows)
        if state == 'experiment-running':
            self.status(state, attempts=[r['id'] for r in rows if r['status']=='running'])
            return
        if lock_busy(self.args.owner / 'continuation.lock'):
            self.status('existing-controller-active')
            return
        mark = fingerprint(self.args.repo,rows,jobs,self.args.prompt)
        if self.receipt.exists():
            prior = json.loads(self.receipt.read_text())
            if (prior.get('returncode') or prior['fingerprint'] == mark) and time.time() < prior['next_review_at']:
                self.status('review-backoff', next_review_at=prior['next_review_at'], last_report=prior.get('report'))
                return
        self.launch(state,mark)

    def main(self):
        with (self.home / 'supervisor.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            while True:
                try:
                    self.tick()
                except Exception as exc:
                    self.status('supervisor-error',error_type=type(exc).__name__,detail=str(exc))
                if self.args.once:
                    return
                time.sleep(self.args.poll_seconds)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('repo','state','owner','home','prompt','schema'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--codex',required=True)
    parser.add_argument('--agent-seconds',type=int,default=2700)
    parser.add_argument('--poll-seconds',type=int,default=20)
    parser.add_argument('--once',action='store_true')
    args=parser.parse_args()
    if any(not getattr(args,k).is_absolute() for k in ('repo','state','owner','home','prompt','schema')):
        parser.error('Use absolute paths')
    Supervisor(args).main()


if __name__=='__main__':main()
