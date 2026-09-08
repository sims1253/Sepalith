import argparse
import fcntl
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import time
from unittest.mock import Mock
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parent))
import supervisor as s


def environment(tmp_path):
    repo=tmp_path/'repo';repo.mkdir()
    subprocess.run(['git','init','-q',str(repo)],check=True)
    subprocess.run(['git','-C',str(repo),'-c','user.name=Test','-c','user.email=test@example.invalid','commit','--allow-empty','-qm','init'],check=True)
    (repo/'docs').mkdir();(repo/'docs/EXPERIMENT-QUEUE.md').write_text('authorized work')
    state=tmp_path/'state';state.mkdir()
    with sqlite3.connect(state/'state.sqlite3') as db:
        db.execute('create table attempts (id text,status text,worker_pid integer,child_pid integer,child_group integer,started real,finished real)')
        db.execute('create table jobs (id text,status text)')
    owner=tmp_path/'owner';owner.mkdir()
    prompt=tmp_path/'prompt';prompt.write_text('test fixture; no real agent')
    schema=tmp_path/'schema';schema.write_text('{}')
    fake=tmp_path/'fake-codex'
    fake.write_text('#!'+sys.executable+'\n'+'''import json,sys
from pathlib import Path
sys.stdin.read()
p=Path(sys.argv[sys.argv.index('--output-last-message')+1])
p.write_text(json.dumps(dict(outcome='progress',summary='fixture completed',next_action='next fixture',retry_after_seconds=0,blockers=[])))
print('fixture')
''');fake.chmod(0o755)
    return argparse.Namespace(repo=repo,state=state,owner=owner,home=tmp_path/'supervisor',prompt=prompt,schema=schema,codex=str(fake),agent_seconds=30,poll_seconds=1,once=True)


def test_live_worker_does_not_spend_agent_call(tmp_path,monkeypatch):
    a=environment(tmp_path)
    with sqlite3.connect(a.state/'state.sqlite3') as db:
        db.execute('insert into attempts values (?,?,?,?,?,?,?)',('live','running',os.getpid(),None,None,time.time(),None))
    sup=s.Supervisor(a);launch=Mock();monkeypatch.setattr(sup,'launch',launch)
    sup.tick();launch.assert_not_called()
    assert json.loads((a.home/'status.json').read_text())['phase']=='experiment-running'


def test_stale_running_row_wakes_recovery(tmp_path,monkeypatch):
    a=environment(tmp_path)
    with sqlite3.connect(a.state/'state.sqlite3') as db:
        db.execute('insert into attempts values (?,?,?,?,?,?,?)',('lost','running',99999999,None,None,time.time(),None))
    sup=s.Supervisor(a);launch=Mock();monkeypatch.setattr(sup,'launch',launch)
    sup.tick();assert launch.call_args.args[0]=='recovery-needed'


def test_reused_worker_pid_is_not_live_attempt():
    assert not s.worker_matches(dict(worker_pid=os.getpid(),started=1))


def test_overdue_live_worker_wakes_review():
    row=dict(status='running',worker_pid=os.getpid(),started=time.time(),runtime_bound_seconds=10)
    assert s.work_state([row],now=row['started']+311)=='deadline-review-needed'


def test_existing_controller_excludes_new_owner(tmp_path,monkeypatch):
    a=environment(tmp_path);sup=s.Supervisor(a);launch=Mock();monkeypatch.setattr(sup,'launch',launch)
    with (a.owner/'continuation.lock').open('w') as f:
        fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);sup.tick()
    launch.assert_not_called()
    assert json.loads((a.home/'status.json').read_text())['phase']=='existing-controller-active'


def test_real_fake_agent_then_new_completion_wakes_again(tmp_path,monkeypatch):
    a=environment(tmp_path);sup=s.Supervisor(a)
    sleep=time.sleep;monkeypatch.setattr(s.time,'sleep',lambda _:sleep(.02))
    sup.tick();first=json.loads(sup.receipt.read_text())
    assert first['returncode']==0 and first['report']['outcome']=='progress'
    sup.tick();assert len(list((a.home/'reviews').iterdir()))==1
    with sqlite3.connect(a.state/'state.sqlite3') as db:
        db.execute('insert into attempts values (?,?,?,?,?,?,?)',('new','succeeded',None,None,None,time.time(),time.time()))
    sup.tick();assert len(list((a.home/'reviews').iterdir()))==2


def test_failure_backoff_survives_queue_changes(tmp_path,monkeypatch):
    a=environment(tmp_path);sup=s.Supervisor(a)
    s.write_json(sup.receipt,dict(fingerprint='changed',returncode=2,next_review_at=time.time()+300,report=None))
    launch=Mock();monkeypatch.setattr(sup,'launch',launch);sup.tick();launch.assert_not_called()


def test_user_hold_prevents_dispatch(tmp_path,monkeypatch):
    a=environment(tmp_path);sup=s.Supervisor(a);(a.home/'HOLD').touch()
    launch=Mock();monkeypatch.setattr(sup,'launch',launch);sup.tick();launch.assert_not_called()


def test_second_supervisor_cannot_start(tmp_path):
    a=environment(tmp_path);sup=s.Supervisor(a)
    with (a.home/'supervisor.lock').open('w') as f:
        fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):sup.main()


def test_invalid_report_backs_off():
    assert s.next_delay(None,0)==300
    assert s.next_delay(dict(outcome='progress',summary='',next_action='x',retry_after_seconds=0),0)==300
