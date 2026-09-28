import datetime as dt,json
import anyscale
cloud='cld_mh2xgqnvkq5squguqzf5wwwqzm'
client=anyscale.Anyscale()._anyscale_client._internal_api_client
api=client
out={'schema':'sepalith.pre04.p4d-account-readonly-probe.v1','observed_at':dt.datetime.now(dt.timezone.utc).isoformat(),'cloud_id':cloud,'target':'p4d.24xlarge','credentials_serialized':False,'mutations':0}
for label,call in [
 ('filtered_gpu_status',lambda:api.get_cloud_gpu_status_api_v2_clouds_cloud_id_gpu_status_get(cloud,instance_type='p4d.24xlarge',_request_timeout=20)),
 ('gpu_status_instance_types',lambda:api.get_cloud_gpu_status_instance_types_api_v2_clouds_cloud_id_gpu_status_instance_types_get(cloud,_request_timeout=20)),
 ('credits',lambda:api.get_credits_v2_api_v2_organization_billing_credits_v2_get(_request_timeout=20))]:
 try:
  x=call();d=x.to_dict() if hasattr(x,'to_dict') else x
  if label=='filtered_gpu_status':
   # Retain only counts/status fields; never instance addresses or user data.
   r=d.get('result',d) if isinstance(d,dict) else {}
   items=r.get('gpus',r.get('items',r.get('results',[]))) if isinstance(r,dict) else []
   out[label]={'status':'returned','result_keys':sorted(r) if isinstance(r,dict) else [],'record_count':len(items) if isinstance(items,list) else None,'target_named':json.dumps(r).count('p4d.24xlarge') if isinstance(r,dict) else 0}
  elif label=='gpu_status_instance_types':
   r=d.get('result',d) if isinstance(d,dict) else {};types=r.get('instance_types',[]) if isinstance(r,dict) else []
   out[label]={'status':'returned','count':len(types),'contains_target':('p4d.24xlarge' in types),'target_count':sum(x=='p4d.24xlarge' for x in types)}
  else:
   # Same narrow balance fields used by prior accepted probe.
   r=d if isinstance(d,dict) else {}
   wanted={k:v for k,v in r.items() if k in ('current_balance_usd','amount_spent_usd','total_granted_usd','expiration_date','expiry_date')}
   out[label]={'status':'returned','fields':wanted}
 except Exception as e:
  out[label]={'status':'error','exception_type':type(e).__name__,'http_status':getattr(e,'status',None)}
print(json.dumps(out,sort_keys=True))
