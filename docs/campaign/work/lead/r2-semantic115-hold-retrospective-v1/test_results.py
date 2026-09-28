#!/usr/bin/env python3
import hashlib,json
from pathlib import Path
p=Path('/mnt/e/sepalith/campaign-20260915/data-work/Semantic115-hold-retrospective-v1')
def sha(q):return hashlib.sha256(q.read_bytes()).hexdigest()
m=json.loads((p/'manifest.json').read_text());rows=[json.loads(x) for x in (p/'classification-ledger.jsonl').open()]
assert m['denominator']==len(rows)==len({x['row_id'] for x in rows})==115
assert sum(m['class_counts'].values())==115 and m['output']['sha256']==sha(p/'classification-ledger.jsonl')
assert sum(x['v2_reprocessed'] for x in rows)==3 and not any(x['v2_infrastructure_failure'] for x in rows if x['v2_reprocessed']) and not any(x['v2_supported'] for x in rows if x['v2_reprocessed'])
assert all(x['resolution'] in ('retained_repair_queue','v2_replay_evidence_hold_no_infrastructure') and x['training_admission'] is False for x in rows)
assert all(not any(k in x for k in ('target','gold','completion')) for x in rows)
print('PASS exact115 IDs/classes; three precedence-risk v2 replays; zero infrastructure/support; all retained; no target/gold')
