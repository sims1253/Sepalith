#!/usr/bin/env python3
"""Read-only hosted-account inventory probe; never serializes credentials or IDs."""
from __future__ import annotations
import argparse,datetime,json
from pathlib import Path

ALLOWED_CALLS=(
 'list_clouds','get_cloud','list_resource_quotas','additional_instance_types_GET',
 'gpu_fleet_instance_types_GET','list_machine_pools_GET','default_compute_GET',
 'plan_status_GET','active_billing_version_GET','cloud_resources_GET','search_cluster_computes',
)

def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def safe_error(error):
 return {'type':type(error).__name__,'status':getattr(error,'status',None),'provider_detail':('hosted_cloud_resources_unavailable' if 'Anyscale-hosted clouds can not be fetched' in str(error) else 'unclassified_error')}
def collect(client):
 clouds=client.list_clouds(count=20).results;api=client._internal_api_client
 result={'schema':'sepalith.anyscale-large-vram-account-evidence.v2','observed_at':now(),'sdk_version':None,'allowed_calls':list(ALLOWED_CALLS),'clouds':[]}
 import anyscale;result['sdk_version']=getattr(anyscale,'__version__','unknown')
 for cloud in clouds:
  full=client.get_cloud(cloud_id=cloud.id)
  entry={'name':cloud.name,'provider':str(cloud.provider),'region':cloud.region,'state':str(cloud.state),'status':str(cloud.status),'compute_stack':str(cloud.compute_stack),'is_anyscale_hosted':bool(full.is_aioa),'is_bring_your_own_resource':bool(full.is_bring_your_own_resource)}
  entry['resource_quota_count']=len(client.list_resource_quotas(cloud_id=cloud.id,max_items=50))
  entry['additional_instance_types']=api.get_cloud_additional_instance_types_api_v2_clouds_cloud_id_additional_instance_types_get(cloud.id).to_dict()
  entry['gpu_fleet_instance_types']=api.get_cloud_gpu_status_instance_types_api_v2_clouds_cloud_id_gpu_status_instance_types_get(cloud.id).to_dict()
  pools=api.list_machine_pools_api_v2_machine_pools_get().to_dict();entry['machine_pool_count']=len(pools.get('results') or [])
  default=api.get_default_compute_config_api_v2_compute_templates_default_cloud_id_get(cloud.id).result
  entry['default_compute']={'head_instance_type':default.head_node_type.instance_type,'worker_instance_types':[x.instance_type for x in default.worker_node_types]}
  computes=client.search_cluster_computes({'cloud_id':cloud.id,'include_anonymous':True,'version':-1}).results
  entry['registered_compute_instance_types']=sorted({x.config.head_node_type.instance_type for x in computes}|{n.instance_type for x in computes for n in x.config.worker_node_types})
  try:api.get_cloud_resources_api_v2_clouds_cloud_id_resources_get(cloud.id);entry['cloud_resources_GET']={'status':'returned'}
  except Exception as error:entry['cloud_resources_GET']={'status':'error',**safe_error(error)}
  result['clouds'].append(entry)
 result['plan_status']=api.get_plan_status_api_v2_organization_billing_plan_status_get().to_dict()
 billing=api.get_active_billing_version_api_v2_organization_billing_active_billing_version_get()
 result['active_billing_version']=str(billing)
 result['price_or_rate_fields_returned']=False
 result['credentials_accessed_or_serialized']=False
 return result

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args()
 if args.output.exists():raise SystemExit('fresh output required')
 import anyscale
 value=collect(anyscale.Anyscale()._anyscale_client);args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(value,indent=2,sort_keys=True)+'\n')
 print(json.dumps({'clouds':len(value['clouds']),'output':str(args.output)}))
if __name__=='__main__':main()
