"""Bounded read-only campaign PID/RSS sampler; reuses existing Windows readings.

No process signals, model/data reads, cache operations, Windows queries, or
persistent monitor. Root must explicitly launch this optional preparation.
"""
from __future__ import annotations
import argparse,datetime,hashlib,json,math,os,time
from pathlib import Path

MEM_FIELDS=('MemTotal','MemFree','MemAvailable','Buffers','Cached','SReclaimable',
            'SUnreclaim','AnonPages','Mapped','Shmem','Dirty','Writeback','SwapTotal','SwapFree')
RSS_FIELDS=('VmRSS','VmHWM','RssAnon','RssFile','RssShmem','VmSize','VmSwap')

def utc_now():return datetime.datetime.now(datetime.timezone.utc).isoformat()

def parse_stat(text):
    close=text.rfind(')')
    if close<0:raise ValueError('malformed proc stat')
    left=text[:close+1];parts=text[close+1:].split()
    if len(parts)<22:raise ValueError('short proc stat')
    return {'pid':int(left.split(' ',1)[0]),'state':parts[0],'ppid':int(parts[1]),
            'start_tick':int(parts[19]),'rss_pages':int(parts[21])}

def parse_kib(text,fields):
    values={}
    for line in text.splitlines():
        key,sep,rest=line.partition(':')
        if sep and key in fields:
            words=rest.split()
            if len(words)!=2 or words[1]!='kB':raise ValueError('memory field lacks kB units')
            value=int(words[0])
            if value<0:raise ValueError('negative memory field')
            values[key]=value
    return values

def read_process(pid,expected_start=None,expected_parent=None,proc_root=Path('/proc')):
    base=proc_root/str(pid)
    try:
        before=parse_stat((base/'stat').read_text())
        if before['pid']!=pid:raise ValueError('proc PID differs')
        if expected_start is not None and before['start_tick']!=expected_start:
            return {'pid':pid,'status':'start_tick_mismatch'}
        if expected_parent is not None and before['ppid']!=expected_parent:
            return {'pid':pid,'status':'parent_changed'}
        status=(base/'status').read_text()
        uid_line=next(x for x in status.splitlines() if x.startswith('Uid:'))
        uid=int(uid_line.split()[1])
        if uid!=os.getuid():return {'pid':pid,'status':'different_uid_not_sampled'}
        rss=parse_kib(status,RSS_FIELDS)
        after=parse_stat((base/'stat').read_text())
        if (before['start_tick'],before['ppid'])!=(after['start_tick'],after['ppid']):
            return {'pid':pid,'status':'identity_changed_during_read'}
        return {'pid':pid,'start_tick':before['start_tick'],'ppid':before['ppid'],
                'state':before['state'],'status':'observed','rss_kib':rss}
    except (FileNotFoundError,ProcessLookupError):return {'pid':pid,'status':'exited'}
    except PermissionError:return {'pid':pid,'status':'permission_denied'}

def process_tree(root_pid,root_start,proc_root=Path('/proc'),max_processes=64,max_thread_files=256):
    root=read_process(root_pid,root_start,proc_root=proc_root)
    if root['status']!='observed' or root.get('state')=='Z':
        return {'root_status':root['status'] if root.get('state')!='Z' else 'zombie','processes':[root],'complete':False}
    queue=[root];records=[];visited=set();thread_files=0;limitations=[]
    while queue and len(records)<max_processes:
        record=queue.pop(0);pid=record['pid']
        if pid in visited:continue
        visited.add(pid);records.append(record)
        if record['status']!='observed':continue
        try:
            tasks=sorted((p for p in (proc_root/str(pid)/'task').iterdir() if p.name.isdecimal()),key=lambda p:int(p.name))
            for task in tasks:
                if thread_files>=max_thread_files:
                    limitations.append('thread_children_scan_cap');break
                thread_files+=1
                try:children=(task/'children').read_text().split()
                except FileNotFoundError:continue
                for child in children:
                    cid=int(child)
                    if cid not in visited:queue.append(read_process(cid,expected_parent=pid,proc_root=proc_root))
        except FileNotFoundError:limitations.append('process_exited_during_children_scan')
        except PermissionError:limitations.append('children_scan_permission_denied')
    if queue:limitations.append('process_count_cap')
    final_root=read_process(root_pid,root_start,proc_root=proc_root)
    final_status='zombie' if final_root.get('state')=='Z' else final_root['status']
    if final_status!='observed':limitations.append('root_changed_during_sample')
    totals={k:sum(r.get('rss_kib',{}).get(k,0) for r in records) for k in RSS_FIELDS}
    return {'root_status':final_status,'processes':records,'rss_sum_kib':totals,
            'complete':not limitations and all(r['status']=='observed' for r in records),
            'limitations':sorted(set(limitations)),'thread_children_files_read':thread_files,
            'rss_sum_limit':'RSS sums may double-count shared pages; no PSS/USS or WDDM attribution is claimed.'}

def latest_windows(path,now_epoch,max_age_seconds):
    try:
        with path.open('rb') as stream:
            stream.seek(max(0,path.stat().st_size-65536));raw=stream.read(65536)
        # Ignore an unfinished trailing line. Guard emits newline-terminated JSON.
        lines=raw.split(b'\n')[:-1]
        record=None
        for line in reversed(lines):
            try:record=json.loads(line);break
            except (json.JSONDecodeError,UnicodeDecodeError):continue
        if not isinstance(record,dict):return {'status':'no_complete_windows_record'}
        observed=datetime.datetime.fromisoformat(record['At'].replace('Z','+00:00'))
        if observed.tzinfo is None:raise ValueError('Windows timestamp lacks timezone')
        age=now_epoch-observed.timestamp()
        selected={'At':record['At']}
        for field in ('AvailableMBytes','CommittedBytes','CommitLimit','PageReadsPersec','PagesInputPersec','PagesOutputPersec'):
            value=record.get(field)
            if type(value) not in (int,float) or not math.isfinite(value) or value<0:raise ValueError('invalid Windows memory counter')
            selected[field]=value
        status='future_timestamp' if age < -2 else ('stale' if age>max_age_seconds else 'observed')
        return {'status':status,'age_seconds':age,'counters':selected,'source':'existing_guard_windows_memory_record',
                'units':{'AvailableMBytes':'Windows available physical MiB','CommittedBytes':'system committed bytes; not process RSS','CommitLimit':'system commit-limit bytes'}}
    except FileNotFoundError:return {'status':'windows_log_not_yet_present'}
    except (ValueError,KeyError,TypeError) as error:return {'status':'invalid_windows_record','error_type':type(error).__name__}

def capture(root_pid,root_start,windows_log,max_age_seconds,proc_root=Path('/proc')):
    started=time.monotonic();epoch=time.time()
    tree=process_tree(root_pid,root_start,proc_root)
    linux=parse_kib((proc_root/'meminfo').read_text(),MEM_FIELDS)
    return {'schema':1,'at':datetime.datetime.fromtimestamp(epoch,datetime.timezone.utc).isoformat(),
            'pid_tree':tree,'linux_memory_kib':linux,
            'windows':latest_windows(windows_log,epoch,max_age_seconds),
            'sample_read_seconds':time.monotonic()-started,
            'comparison_limit':'Linux and Windows timestamps are explicit and may differ. Linux MemAvailable includes reclaimable guest memory; it is not a Windows physical-availability reading.'}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root-pid',type=int,required=True)
    parser.add_argument('--root-start-tick',type=int,required=True)
    parser.add_argument('--windows-log',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--seconds',type=int,default=600)
    parser.add_argument('--interval-seconds',type=float,default=15)
    args=parser.parse_args()
    if not 1<=args.seconds<=1800 or not 10<=args.interval_seconds<=60:raise ValueError('duration or interval outside bounded range')
    if args.root_pid<=1 or args.root_start_tick<=0:raise ValueError('explicit live root PID/start tick required')
    if not args.output.is_absolute() or args.output.exists():raise ValueError('output must be a new absolute directory')
    if args.windows_log.name!='host-memory.jsonl':raise ValueError('reuse the existing guard host-memory.jsonl')
    launch_path=args.windows_log.parent/'launch.json'
    launch_raw=launch_path.read_bytes();launch=json.loads(launch_raw)
    if launch.get('guard_pid')!=args.root_pid:raise ValueError('Windows guard launch identity differs')
    initial=read_process(args.root_pid,args.root_start_tick)
    if initial['status']!='observed' or initial['state']=='Z':raise ValueError('root PID/start tick is not live')
    if hasattr(os,'sched_setaffinity'):os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
    args.output.mkdir(parents=True,exist_ok=False)
    start=time.monotonic();count=0;total_read=0;reason='bounded_duration_complete'
    (args.output/'configuration.json').write_text(json.dumps({'schema':1,'at':utc_now(),'root_pid':args.root_pid,
        'root_start_tick':args.root_start_tick,'guard_launch_sha256':hashlib.sha256(launch_raw).hexdigest(),
        'windows_log':str(args.windows_log),'seconds':args.seconds,'interval_seconds':args.interval_seconds,
        'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'scope':'Read-only optional root-launched sampler. Reuses guard Windows queries. No signals, cache operations, model/data reads, process command lines or environments.'},indent=2)+'\n')
    try:
        with (args.output/'samples.jsonl').open('x') as output:
            while time.monotonic()-start<args.seconds:
                sample=capture(args.root_pid,args.root_start_tick,args.windows_log,2*args.interval_seconds+10)
                output.write(json.dumps(sample,allow_nan=False,separators=(',',':'))+'\n');output.flush()
                count+=1;total_read+=sample['sample_read_seconds']
                if sample['pid_tree']['root_status']!='observed':reason='root_identity_no_longer_live';break
                remaining=args.seconds-(time.monotonic()-start)
                if remaining<=0:break
                time.sleep(min(args.interval_seconds,remaining))
    except BaseException as error:
        reason="sampler_error_"+type(error).__name__
        raise
    finally:
        (args.output/'terminal.json').write_text(json.dumps({'at':utc_now(),'status':reason,'samples':count,
            'seconds':time.monotonic()-start,'sample_read_seconds_total':total_read,'root_process_controlled':False,
            'scope':'Sampler completion only; not proof of root process cleanup, host safety, or model acceptance.'},indent=2)+'\n')

if __name__=='__main__':main()
