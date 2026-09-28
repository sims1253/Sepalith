#!/usr/bin/env python3
"""Reject topology changes that cannot faithfully consume checkpoint-90 RNG/cursor state."""
import argparse,json
from pathlib import Path
from transport_common import require,sha256
MANIFEST_SHA='d820a0f5cd3c790f470212406c841fb3a34924131ac5fc098f2a22c1b2db9252'
def gate(checkpoint,world_size,recipe,continuation_admission):
 root=Path(checkpoint);require(sha256(root/'campaign-manifest.json')==MANIFEST_SHA,'checkpoint manifest differs');require(world_size==1,'checkpoint90 has single-process rng_state.pth; distributed resume requires separately admitted rank RNG/topology migration');require((root/'rng_state.pth').is_file() and not any(root.glob('rng_state_*.pth')),'checkpoint RNG topology differs');require(Path(recipe).is_file() and Path(continuation_admission).is_file(),'bound recipe/continuation admission absent');return {'status':'pass','world_size':1,'resume_checkpoint':str(root),'recipe':str(Path(recipe)),'continuation_admission':str(Path(continuation_admission)),'trainer_argv':['full_weight_cpt_trainer.py','run','--recipe',str(Path(recipe)),'--resume',str(root),'--continuation-admission',str(Path(continuation_admission))]}
def main():
 p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--world-size',type=int,required=True);p.add_argument('--recipe',type=Path,required=True);p.add_argument('--continuation-admission',type=Path,required=True);a=p.parse_args();print(json.dumps(gate(a.checkpoint,a.world_size,a.recipe,a.continuation_admission),sort_keys=True))
if __name__=='__main__':main()
