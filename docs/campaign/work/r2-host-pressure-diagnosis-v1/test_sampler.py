"""Synthetic proc/log checks plus one read of this CPU test process."""
import datetime,json,os,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import sample_host_memory as s
HERE=Path(__file__).resolve().parent

def stat(pid,start,parent=1,state='S',comm='python (worker)'):
    parts=[state,str(parent)]+['0']*20;parts[19]=str(start);parts[21]='16'
    return str(pid)+' ('+comm+') '+' '.join(parts)+'\n'

def process(root,pid,start,parent=1,children='',uid=None,state='S'):
    p=root/str(pid);(p/'task'/str(pid)).mkdir(parents=True)
    (p/'stat').write_text(stat(pid,start,parent,state))
    (p/'status').write_text(f'Uid:\t{os.getuid() if uid is None else uid}\t0\t0\t0\nVmRSS:\t100 kB\nRssAnon:\t60 kB\nRssFile:\t30 kB\nRssShmem:\t10 kB\nVmHWM:\t120 kB\n')
    (p/'task'/str(pid)/'children').write_text(children)

class SamplerChecks(unittest.TestCase):
    def test_stat_parentheses_and_start_tick(self):
        p=s.parse_stat(stat(10,1234,9,comm='a) b (c)'))
        self.assertEqual((p['pid'],p['ppid'],p['start_tick']),(10,9,1234))

    def test_owned_tree_rss_and_thread_children(self):
        with tempfile.TemporaryDirectory(dir=HERE) as d:
            p=Path(d);process(p,10,100,children='11');process(p,11,200,parent=10)
            other=p/'10/task/19';other.mkdir();(other/'children').write_text('12')
            process(p,12,201,parent=10)
            result=s.process_tree(10,100,p)
            self.assertTrue(result['complete']);self.assertEqual(len(result['processes']),3)
            self.assertEqual(result['rss_sum_kib']['VmRSS'],300)

    def test_pid_reuse_rejected_before_rss(self):
        with tempfile.TemporaryDirectory(dir=HERE) as d:
            p=Path(d);process(p,10,200)
            r=s.read_process(10,100,proc_root=p)
            self.assertEqual(r,{'pid':10,'status':'start_tick_mismatch'})

    def test_reparented_and_other_uid_not_sampled(self):
        with tempfile.TemporaryDirectory(dir=HERE) as d:
            p=Path(d);process(p,10,100,children='11 12');process(p,11,200,parent=9)
            process(p,12,201,parent=10,uid=os.getuid()+1)
            r=s.process_tree(10,100,p)
            self.assertFalse(r['complete']);self.assertEqual(r['rss_sum_kib']['VmRSS'],100)
            self.assertEqual({x['status'] for x in r['processes']},{'observed','parent_changed','different_uid_not_sampled'})

    def test_identity_race_and_missing_process(self):
        with tempfile.TemporaryDirectory(dir=HERE) as d:
            p=Path(d);process(p,10,100)
            first=s.parse_stat(stat(10,100));second={**first,'start_tick':101}
            with patch.object(s,'parse_stat',side_effect=[first,second]):
                self.assertEqual(s.read_process(10,100,proc_root=p)['status'],'identity_changed_during_read')
            self.assertEqual(s.process_tree(20,100,p)['root_status'],'exited')

    def test_zombie_and_scan_cap(self):
        with tempfile.TemporaryDirectory(dir=HERE) as d:
            p=Path(d);process(p,10,100,state='Z')
            self.assertEqual(s.process_tree(10,100,p)['root_status'],'zombie')
            (p/'10/stat').write_text(stat(10,100));r=s.process_tree(10,100,p,max_thread_files=0)
            self.assertFalse(r['complete']);self.assertIn('thread_children_scan_cap',r['limitations'])

    def test_windows_completed_line_staleness_and_invalid_numbers(self):
        with tempfile.TemporaryDirectory(dir=HERE) as d:
            p=Path(d)/'host-memory.jsonl';now=1000000000
            row={'At':datetime.datetime.fromtimestamp(now,datetime.timezone.utc).isoformat(),
                 'AvailableMBytes':10000,'CommittedBytes':1,'CommitLimit':2,'PageReadsPersec':0,'PagesInputPersec':0,'PagesOutputPersec':0}
            p.write_text(json.dumps(row)+'\n'+ '{"At":')
            self.assertEqual(s.latest_windows(p,now+1,40)['status'],'observed')
            self.assertEqual(s.latest_windows(p,now+41,40)['status'],'stale')
            self.assertEqual(s.latest_windows(p,now-3,40)['status'],'future_timestamp')
            row['AvailableMBytes']=float('nan');p.write_text(json.dumps(row)+'\n')
            self.assertEqual(s.latest_windows(p,now,40)['status'],'invalid_windows_record')

    def test_meminfo_units(self):
        self.assertEqual(s.parse_kib('Cached: 1024 kB\nAnonPages: 2048 kB\n',s.MEM_FIELDS),{'Cached':1024,'AnonPages':2048})
        with self.assertRaises(ValueError):s.parse_kib('Cached: 10 MB',s.MEM_FIELDS)

    def test_one_live_self_pid_read_only(self):
        pid=os.getpid();expected=s.parse_stat((Path('/proc')/str(pid)/'stat').read_text())['start_tick']
        r=s.read_process(pid,expected)
        self.assertEqual(r['status'],'observed');self.assertGreater(r['rss_kib']['VmRSS'],0)

if __name__=='__main__':unittest.main(verbosity=2)
