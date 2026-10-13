#!/usr/bin/env python3
"""Reproduce the checkpoint I/O pass audit from frozen source and telemetry."""
import hashlib,json,statistics,time
from pathlib import Path
PLAN=Path('/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb')
SRC=PLAN/'docs/campaign/work/lead/r2-selected330-recovery-preparation-v2/source/experiments/training'
TELEMETRY=Path('/mnt/e/sepalith/campaign-20260915/checkpoints/SFT11-selected330-recovery-cadence24-v2/telemetry.jsonl')
PINS={'campaign_checkpoint.py':'75472c65dc44479ca6090cc6be86a2e196ccbbda684e20e3a45cef6eb96acaa8','native_checkpoint_publish.py':'220115bef24c240b2425fb222a7e6435787f352fcd08e3ae0ec9022c773baa1e','full_weight_cpt_trainer.py':'5184354d974cc9bedfbc055f1d6b4c7091582e50a6d2b68c677c6a43ddfc79ee'}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 for n,h in PINS.items():assert sha(SRC/n)==h
 c=(SRC/'campaign_checkpoint.py').read_text();p=(SRC/'native_checkpoint_publish.py').read_text();t=(SRC/'full_weight_cpt_trainer.py').read_text()
 assert 'flush_files(directory)' in c and 'files = inventory(directory)' in c
 assert 'copied[name]=copy_hash(source/rel,temporary/rel,expected)' in p and 'out.flush();os.fsync(out.fileno())' in p
 assert "verify_checkpoint(terminal, identity(recipe), require_full=True" in t
 rows=[json.loads(x) for x in TELEMETRY.read_text().splitlines()];logs=[x for x in rows if x.get('event')=='log' and 'loss' in x.get('logs',{})]
 assert [x['step'] for x in logs]==list(range(331,355))
 intervals=[b['at']-a['at'] for a,b in zip(logs,logs[1:])];seal=next(x for x in rows if x.get('event')=='native_seal_end_publish_start')
 end=next((x for x in rows if x.get('event')=='durable_E_publish_end'),None)
 publication={'start_at_epoch':seal['at'],'terminal_observed':end is not None}
 publication['seconds']=end['at']-seal['at'] if end else max(0,time.time()-seal['at']);publication['measurement']='exact' if end else 'lower_bound'
 result={'schema':'sepalith.sft11.checkpoint-publication-throughput-audit.v1','status':'source_and_live_metadata_audited_no_payload_reads','source_pins':PINS,'checkpoint_bytes':seal['checkpoint_bytes'],'compute':{'updates':24,'first_to_last_log_seconds':logs[-1]['at']-logs[0]['at'],'mean_step_interval_seconds':statistics.mean(intervals),'median_step_interval_seconds':statistics.median(intervals)},'publication':publication,'passes':{'native_write':1,'native_full_read_before_root_acceptance':2,'durable_E_write':1,'durable_E_full_read_inside_trainer_terminal':1,'additional_root_acceptance_native_full_read':1,'additional_root_acceptance_E_full_read':1},'notes':['flush_files fsyncs each native file descriptor but does not intentionally read payload bytes','seal inventory hashes every native payload once','publisher reads/hashes native once while writing and fsyncing E temporary files','atomic E visibility uses temporary directory, manifest fsync, directory fsync, rename, parent fsync','terminal verify inventories/hashes E once','admitted root verifier independently hashes native and E once each']}
 out=Path(__file__).with_name('result.json');out.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n');print(json.dumps(result,sort_keys=True))
if __name__=='__main__':main()
