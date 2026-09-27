#!/usr/bin/env python3
import argparse, json
from pathlib import Path
ap=argparse.ArgumentParser();ap.add_argument('--attempt',type=Path,required=True);a=ap.parse_args()
manifest=json.loads((a.attempt/'manifest.json').read_text());rows=[json.loads(x) for x in (a.attempt/'sample-ledger.jsonl').open()]
normalized={};licenses={}
for r in rows:
 p=Path(r['normalized_source_path']);normalized[str(p)]={'path':str(p),'bytes':p.stat().st_size,'sha256':r['normalized_source_sha256']}
 p=Path(r['license_path']);licenses[str(p)]={'path':str(p),'bytes':p.stat().st_size,'sha256':r['license_sha256']}
o={'schema':'sepalith.dat10.sourcewalk_independent_replay_inventory.v1','rows':len(rows),'raw_source_files':manifest['source_files'],'normalized_source_files':list(sorted(normalized.values(),key=lambda x:x['path'])),'license_files':list(sorted(licenses.values(),key=lambda x:x['path'])),'all_rows_have_exact_paths':all(r.get('raw_source_path') and r.get('raw_source_line') and r.get('normalized_source_path') and r.get('license_path') for r in rows),'source_or_target_text_written':False}
(a.attempt/'source-inventory.json').write_text(json.dumps(o,indent=2,sort_keys=True)+'\n')
