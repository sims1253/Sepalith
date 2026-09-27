#!/usr/bin/env python3
"""Generate a reusable parent+corpus native stage spec from a bound recipe."""
import argparse,hashlib,json
from pathlib import Path
def require(v,m):
 if not v:raise ValueError(m)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def prepare(recipe_path,recipe_sha,native_root,output):
 recipe_path=Path(recipe_path);require(sha(recipe_path)==recipe_sha,'bound recipe differs');r=json.loads(recipe_path.read_text())
 parent=Path(r['parent']['path']);pm=parent/'campaign-manifest.json';require(sha(pm)==r['transition']['source_checkpoint']['manifest_sha256'],'parent checkpoint manifest differs');pv=json.loads(pm.read_text())
 for name,value in r['parent']['files'].items():require(pv['files'][name]['sha256']==value,'parent file identity differs')
 cache=Path(r['cohort']['streaming_cache']['path']);cm=cache/'manifest.json';require(sha(cm)==r['cohort']['streaming_cache']['manifest_sha256'],'cache manifest differs');cv=json.loads(cm.read_text())
 require(cv['source']['rows']['sha256']==r['cohort']['rows']['sha256'] and cv['source']['draw_schedule']['sha256']==r['cohort']['draw_schedule']['sha256'],'cache source identities differ')
 rows=Path(r['cohort']['rows']['path']);schedule=Path(r['cohort']['draw_schedule']['path']);require(rows.is_file()and schedule.is_file(),'corpus input missing')
 value={'schema':'sepalith.sft11.native-stage-spec.v1','status':'prepared_from_bound_recipe_no_copy','native_root':str(Path(native_root).resolve()),'max_native_bytes':32*1024**3,'min_free_after_bytes':70*1024**3,'objects':[
 {'name':'cache','kind':'manifest_tree','source':str(cache.resolve()),'manifest_name':'manifest.json','manifest':{'path':str(cm.resolve()),'sha256':sha(cm)}},
 {'name':'rows','kind':'file','source':str(rows.resolve()),'destination_name':rows.name,'bytes':rows.stat().st_size,'sha256':r['cohort']['rows']['sha256']},
 {'name':'schedule','kind':'file','source':str(schedule.resolve()),'destination_name':schedule.name,'bytes':schedule.stat().st_size,'sha256':r['cohort']['draw_schedule']['sha256']}],
 'binding':{'canonical_bound_recipe':str(recipe_path.resolve()),'canonical_bound_recipe_sha256':recipe_sha,'canonical_parent_manifest_sha256':sha(pm),'resume_checkpoint_policy':'verified separately and used as model bootstrap at every launch; reusable data base is never recopied'}}
 Path(output).write_text(json.dumps(value,indent=2,sort_keys=True)+'\n');return value
def main():
 p=argparse.ArgumentParser();p.add_argument('--recipe',required=True);p.add_argument('--recipe-sha256',required=True);p.add_argument('--native-root',required=True);p.add_argument('--output',required=True);a=p.parse_args();print(json.dumps(prepare(a.recipe,a.recipe_sha256,a.native_root,a.output),sort_keys=True));return 0
if __name__=='__main__':raise SystemExit(main())
