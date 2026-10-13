#!/usr/bin/env python3
"""Launch one bounded semantic review queue for a committed shards-6+ frontier.

The reviewed v3 worker is reused byte-for-byte. This wrapper owns only intake
closure, launch/terminal telemetry, and a fresh bounded output directory.
"""
from __future__ import annotations
import datetime, hashlib, json, os, pathlib, subprocess, sys, time
PACKET=pathlib.Path(__file__).resolve().parent
PLAN=pathlib.Path('/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb')
V3=PLAN/'docs/campaign/work/lead/r2-semantic-queue-root-launch-v3'
REPLAY=pathlib.Path('/mnt/e/sepalith/campaign-20260915/data-work/Sourcewalk-independent-replay-v3/full-01')
OUTPUT=pathlib.Path('/mnt/e/sepalith/campaign-20260915/data-work/Sourcewalk-semantic-streaming-queue-shards6plus-v1')
SELECTED=[6,7,8,9,10,11]
REPLAY_PIDS=[1689900,1689901,1691118]

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()
def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value,indent=2,sort_keys=True)+'\n')
    tmp.replace(path)
def require(ok,msg):
    if not ok: raise RuntimeError(msg)
def file_stable(path,expected_sha,expected_bytes):
    before=path.stat()
    raw=path.read_bytes()
    after=path.stat()
    require(before.st_size==expected_bytes and len(raw)==expected_bytes,'size changed: '+str(path))
    require((before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns)==(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns),'file changed during preflight: '+str(path))
    require(hashlib.sha256(raw).hexdigest()==expected_sha,'hash changed: '+str(path))
def preflight():
    sm=json.loads((V3/'source-manifest.json').read_text())
    for item in sm['files']:
        p=V3/item['path']; require(p.stat().st_size==item['bytes'],'review source size changed')
        require(sha(p)==item['sha256'],'review source hash changed: '+str(p))
    ip=REPLAY/'index/manifest.json'; idx=json.loads(ip.read_text())
    require(idx.get('schema')=='sepalith.dat10.sourcewalk_raw_index.v3' and idx.get('status')=='complete','replay index not complete v3')
    entries={x['shard']:x for x in idx['index_files']}
    require(len(entries)==len(idx['index_files']),'duplicate index entries')
    snapshots=[]
    for n in SELECTED:
        require(n in entries,'selected shard absent from index')
        e=entries[n]; p=pathlib.Path(e['path']); require(p.is_file(),'selected index absent')
        file_stable(p,e['sha256'],e['bytes'])
        rp=REPLAY/'shards'/f'shard-{n:04d}'/'receipt.json'; require(rp.is_file(),'selected receipt absent')
        rx=json.loads(rp.read_text()); require(rx.get('status')=='complete' and rx.get('shard')==n,'selected receipt incomplete')
        require(rx.get('rows')==e.get('rows'),'index/receipt rows differ')
        # A live replay retains only its owner lock here. Refuse if it has a
        # selected index file open, avoiding a read/write race.
        for pid in REPLAY_PIDS:
            fdroot=pathlib.Path('/proc')/str(pid)/'fd'
            if not fdroot.exists(): continue
            for fd in fdroot.iterdir():
                try: target=pathlib.Path(os.path.realpath(fd))
                except OSError: continue
                require(target != p,'live replay FD holds selected shard: '+str(target))
        snapshots.append({'shard':n,'rows':e['rows'],'bytes':e['bytes'],'sha256':e['sha256'],'receipt_sha256':sha(rp)})
    # Ensure no stale output can be treated as a resumable child.
    require(not OUTPUT.exists(),'fresh semantic output already exists')
    return {'index_path':str(ip),'index_sha256':sha(ip),'index_status':idx.get('status'),'selected':snapshots,
            'replay_processes':REPLAY_PIDS,'replay_process_owner':'existing controller, not restarted','at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
def main():
    pre=preflight()
    OUTPUT.mkdir(parents=True,exist_ok=False)
    log=OUTPUT.parent/'Sourcewalk-semantic-streaming-queue-shards6plus-v1.log'
    command=['timeout','--signal=TERM','--kill-after=30s','2100','ionice','-c3','nice','-n','10','taskset','-c','2,3','/usr/bin/python3','-B',str(V3/'source/run_streaming_semantic_queue.py'),'--replay-root',str(REPLAY),'--output',str(OUTPUT),'--shards',','.join(map(str,SELECTED)),'--max-workers','2']
    env=os.environ.copy();env.update(PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONPATH='/mnt/e/sepalith/campaign-20260915/data-work/Sourcewalk-independent-replay-v3/parser-dependencies-root-v1')
    started=datetime.datetime.now(datetime.timezone.utc).isoformat(); t=time.monotonic()
    with log.open('xb') as stream:
        child=subprocess.Popen(command,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        launch={'schema':'sepalith.dat10.semantic_shards6plus_launch.v1','status':'running','training_admission':False,'started_at_utc':started,'controller_pid':os.getpid(),'timeout_pid':child.pid,'command':command,'environment':{k:env[k] for k in ('PYTHONNOUSERSITE','CUDA_VISIBLE_DEVICES','OMP_NUM_THREADS','PYTHONPATH')},'preflight':pre,'selected_shards':SELECTED,'time_limit_seconds':2100,'output':str(OUTPUT),'log':str(log),'source_reuse_packet':str(V3)}
        write(PACKET/'launch.json',launch)
        write(PACKET/'status.json',{**launch,'status':'running','updated_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})
        code=child.wait()
    terminal={'schema':'sepalith.dat10.semantic_shards6plus_terminal.v1','status':'terminal','training_admission':False,'finished_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'exit_code':code,'elapsed_seconds':time.monotonic()-t,'launch':str(PACKET/'launch.json'),'output':str(OUTPUT),'log':str(log)}
    sm=OUTPUT/'streaming-manifest.json'
    if sm.is_file(): terminal['streaming_manifest']={'path':str(sm),'bytes':sm.stat().st_size,'sha256':sha(sm)}
    write(PACKET/'terminal.json',terminal)
    write(PACKET/'status.json',{**terminal,'status':'terminal','updated_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})
    return code
if __name__=='__main__': raise SystemExit(main())
