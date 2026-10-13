import anyscale,json,datetime,pathlib
w=pathlib.Path(__file__).parent;a=anyscale.Anyscale()._anyscale_client._internal_api_client;r={'at':datetime.datetime.now(datetime.timezone.utc).isoformat()}
for k,f,args in [('fleet',a.get_cloud_gpu_status_api_v2_clouds_cloud_id_gpu_status_get,['cld_mh2xgqnvkq5squguqzf5wwwqzm']),('credits',a.get_credits_v2_api_v2_organization_billing_credits_v2_get,[])]:
 try:
  x=f(*args,_request_timeout=20).to_dict();x=x.get('result',x)
  if k=='fleet':x={z:x.get(z) for z in ['snapshot_time','node_rollup','groups']};x['groups_count']=len(x.pop('groups') or [])
  r[k]=x
 except Exception as e:r[k]={'error_type':type(e).__name__}
(w/'account.json').write_text(json.dumps(r,indent=2,default=str)+'\n');print(json.dumps(r,indent=2,default=str))
