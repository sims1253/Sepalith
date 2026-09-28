"""Thin wrapper around the reviewed campaign_launch deadline constructor."""
import argparse, json, os, sys
from pathlib import Path
import campaign_launch
from campaign_checkpoint import write_json,digest
def main():
 p=argparse.ArgumentParser();p.add_argument('recipe',type=Path);p.add_argument('--receipt',type=Path,required=True);a=p.parse_args()
 r=json.loads(a.recipe.read_text());assert campaign_launch.training_entrypoint(r).name=='campaign_task_sft.py'
 assert not a.receipt.exists()
 command=[sys.executable,str(Path(__file__).with_name('cloud_train.py')),str(a.recipe.resolve())]
 argv,supervision=campaign_launch.supervised_command(r,command)
 write_json(a.receipt,{'pid':os.getpid(),'recipe_sha256':digest(a.recipe),'argv':argv,'entry_wrapper':'cloud_train.py; delegates actual campaign_task_sft.run',**supervision})
 env=dict(os.environ,SEPALITH_CAMPAIGN_SOFT_DEADLINE=supervision['soft_deadline'],SEPALITH_CAMPAIGN_HARD_DEADLINE=supervision['hard_deadline'])
 os.execve(campaign_launch.TIMEOUT,argv,env)
if __name__=='__main__':main()
