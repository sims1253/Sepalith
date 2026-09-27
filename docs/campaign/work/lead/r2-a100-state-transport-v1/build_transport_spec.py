#!/usr/bin/env python3
import argparse,json
from pathlib import Path
from transport_common import atomic_json,load_checkpoint_manifest,validate_small_state
REPO='scholzmx/sepalith-lora';MANIFEST_SHA='d820a0f5cd3c790f470212406c841fb3a34924131ac5fc098f2a22c1b2db9252'
def build(checkpoint):
 m,stats=load_checkpoint_manifest(checkpoint,MANIFEST_SHA);state=validate_small_state(checkpoint,m);prefix='full-weight-state/'+MANIFEST_SHA[:24]+'/checkpoint-90'
 files={n:{**row,'source_mode':stats[n][4],'restore_mode':0o600} for n,row in sorted(m['files'].items())}
 files['campaign-manifest.json']={'bytes':(Path(checkpoint)/'campaign-manifest.json').stat().st_size,'sha256':MANIFEST_SHA,'source_mode':stats['campaign-manifest.json'][4],'restore_mode':0o600}
 return {'schema':'sepalith.pre04.full_weight_transport_spec.v1','status':'prepared_upload_not_authorized','repo':REPO,'repo_type':'model','private_required':True,'prefix':prefix,'checkpoint_path':str(Path(checkpoint).resolve()),'campaign_manifest_sha256':MANIFEST_SHA,'step':90,'checkpoint_kind':'full_weights','files':files,'total_bytes':sum(x['bytes'] for x in files.values()),'state':state,'publication':'files first; closure.json committed last; restore requires exact closure revision and SHA','distributed_resume':{'world_size':1,'reason':'source checkpoint contains rng_state.pth only; Trainer distributed resume requires rng_state_<rank>.pth'}}
def main():
 p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();atomic_json(a.output,build(a.checkpoint));print(json.dumps({'status':'prepared','path':str(a.output)}))
if __name__=='__main__':main()
