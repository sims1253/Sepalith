"""Metadata-only matched-panel review after root-owned evaluation."""
import argparse,hashlib,json,math
from pathlib import Path
EXPECTED={'anchor2k':(499,661360),'8k':(20,126464),'16k':(6,69137)}
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def validate_panel(panel,label,binding_sha):
 rows,tokens=EXPECTED[label];assert panel['step']==450 and panel['binding_sha256']==binding_sha
 assert panel['denominators']['validation_rows']==rows and panel['denominators']['validation_loss_tokens']==tokens
 metrics=panel['row_metrics'];assert len(metrics)==rows and len({x['row_id'] for x in metrics})==rows and sum(x['loss_tokens'] for x in metrics)==tokens
 assert math.isfinite(panel['metrics']['mean_causal_nll']);return panel['metrics']['mean_causal_nll']
def review(binding_path,output):
 binding_sha=sha(binding_path);result=json.loads((output/'result.json').read_text());assert result['status']=='evaluations_complete' and result['binding_sha256']==binding_sha and result['training_performed'] is False and result['promotion_authorized'] is False
 values={label:validate_panel(json.loads((output/f'{label}.json').read_text()),label,binding_sha) for label in EXPECTED}
 print(json.dumps({'schema':'sepalith.sft11.cpt450-matched-review.v1','status':'metrics_verified_no_promotion','checkpoint_step':450,'binding_sha256':binding_sha,'mean_causal_nll':values},sort_keys=True))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--binding',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();review(a.binding,a.output)
