"""Analyze retained host/guest observations without touching training files."""
import datetime,hashlib,json
from pathlib import Path
from sample_host_memory import MEM_FIELDS,parse_kib
HERE=Path(__file__).resolve().parent

def read(name):return json.loads((HERE/name).read_text())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    host=[json.loads(x) for x in (HERE/'smoke-host-memory.jsonl').read_text().splitlines()]
    low=[json.loads(x) for x in (HERE/'smoke-low-memory-details.jsonl').read_text().splitlines()]
    pairs=[]
    for sample in low:
        win=next(x for x in host if x['At']==sample['at'])
        linux=parse_kib(sample['linux_memory'],MEM_FIELDS)
        pairs.append({'at':sample['at'],'windows_available_mib':win['AvailableMBytes'],
            'windows_committed_bytes':win['CommittedBytes'],'linux_kib':linux})
    after=read('root-host-after-reclaim.json');after_linux=parse_kib(after['linux_memory'],MEM_FIELDS)
    final=pairs[-1];before_linux=final['linux_kib']
    current=read('current-linux-memory.json')
    result={'status':'bounded_independent_diagnosis_with_attribution_limits',
        'paired_low_memory_observations':pairs,
        'root_after_reclaim_observation':{'at':after['at'],'query_exit':after['query_exit'],'windows':after['windows'],'linux_kib':after_linux},
        'observed_interval_changes':{
            'windows_available_mib':after['windows']['AvailableMBytes']-final['windows_available_mib'],
            'windows_committed_bytes':after['windows']['CommittedBytes']-final['windows_committed_bytes'],
            'linux_cached_kib':after_linux['Cached']-before_linux['Cached'],
            'linux_anon_pages_kib':after_linux['AnonPages']-before_linux['AnonPages'],
            'linux_cached_plus_sreclaimable_after_kib':after_linux['Cached']+after_linux['SReclaimable'],
            'attribution_limit':'This retained interval spans trainer shutdown and cache maintenance. It cannot assign the whole change to either operation.'},
        'root_reclaim_command_outcome':read('root-post-stop-reclaim.json'),
        'prior_owned_model_cache_operation':read('smoke-post-load-cache-release.json'),
        'current_linux_only_observation':{'at':current['at'],'linux_kib':parse_kib(current['linux_memory'],MEM_FIELDS),'windows_sample':'not collected; no extra Windows query during this diagnosis'},
        'findings':[
            'Windows physical availability breached the configured floor; Linux MemAvailable remained high because it is a different guest-level estimate.',
            'Large Linux Cached values and separately observed recovery after reclamation support a substantial reclaimable-cache contribution to residual pressure.',
            'The retained after observation has Cached1,858,592KiB and SReclaimable3,775,856KiB; their sum is approximately5.37GiB. Do not label that sum Cached alone.',
            'A15-second WSL-wrapper timeout does not prove command completion. The root receipt correctly records a separately observed effect and no command-success claim.',
            'Only one model file received the post-load advisory eviction. That receipt does not establish that other campaign file pages or all WSL resident pages were reclaimed.',
            'No campaign PID/starttick/RSS time series or Windows WDDM-specific host-memory counters were retained. Exact process-RSS versus WDDM versus WSL-cache attribution is unresolved.',
            'Zero new driver events and finite recorded CUDA allocation peaks do not prove that WDDM used no host RAM.',
        ],
        'recommendations':[
            'Keep the current8,192MiB soft floor and4,096MiB hard floor for the retry; these observations do not justify lowering them.',
            'Keep independent model/data hash and token-file validation paused during actual load/profile. Complete required identity checks; do not skip them to save cache.',
            'Prefer advisory eviction of exact already-verified campaign files after all owned readers/mappings close, at an admitted pre-update boundary. Candidate scope is the owned model plus known frozen train/validation inputs; no broad path scan or global cache-clear loop.',
            'Measure Windows physical availability after advisory eviction. POSIX_FADV_DONTNEED and compaction receipts alone do not establish physical recovery.',
            'Use the optional read-only sampler on the exact root-owned guard tree to correlate per-process anonymous/file/shared RSS, Linux Cached/Dirty/Writeback and existing Windows readings across load, update and checkpoint phases.',
            'If pressure follows anonymous process RSS while cache stays stable, investigate live allocations/data representations rather than further cache eviction. Current source retains normalized/materialized Python rows and an Arrow Dataset; the larger corpus needs a separately measured host-memory gate.',
            'Global guest cache clear/compaction remains root-owned exceptional maintenance outside measured optimization; the timed-out prior wrapper must remain a partial outcome.'
        ],
        'sampler_limits':['No independent Windows query; uses guard record with explicit timestamp/age and rejects stale/future records as measurement inputs.',
            'Only verified guard descendants, capped at64processes/256thread-child files per sample; incomplete scans are explicit.',
            'RSS sums can double-count shared pages; no PSS/USS or WDDM attribution.',
            'Default600seconds at15-second intervals, hard bounds1..1800seconds/10..60-second intervals; no monitor launched by worker.'],
        'worker_actions':{'GPU_calls':0,'process_signals':0,'cache_operations':0,'training_or_model_content_reads':0,'Windows_queries':0,'persistent_monitor_launches':0},
        'source_sha256':sha(Path(__file__))}
    (HERE/'diagnosis.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'status':result['status'],'changes':result['observed_interval_changes']}))

if __name__=='__main__':main()
