from pathlib import Path
import sys,json
from huggingface_hub import HfApi,hf_hub_download,get_token
w=Path(__file__).resolve().parent
sys.path.insert(0,str(w/'payload'))
from artifact_upload import upload_sentinel
b=json.loads((w/'payload/binding-preparation.json').read_text())
b['artifact_prefix']+='/root-preflight'
r=w/'private-persistence-preflight';r.mkdir(exist_ok=False)
t=get_token();assert t
api=HfApi(token=t)
def download(*args,**kwargs):return hf_hub_download(*args,token=t,**kwargs)
result=upload_sentinel(r,b,api,download)
print(json.dumps(result))
