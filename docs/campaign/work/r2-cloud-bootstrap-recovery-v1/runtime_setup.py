"""Executed only on an admitted cloud host; anonymous public model staging."""
import importlib.metadata, json, os, sys, sysconfig
from pathlib import Path
from cloud_entry import HERE, MODEL_PINS, REVISION, require, sha, write
def main():
 run=Path(sys.argv[1]);require(sys.version_info[:3]==(3,10,19),'managed Python patch differs')
 headers=Path(sysconfig.get_path('include'))/'Python.h';require(headers.is_file(),'Python development headers missing')
 pins=json.loads((HERE/'package-pins.json').read_text());actual={k:importlib.metadata.version(k) for k in pins};require(actual==pins,'installed package pins differ')
 import torch
 require(torch.cuda.is_available() and torch.cuda.device_count()==1,'single cloud GPU missing')
 props=torch.cuda.get_device_properties(0)
 require('A10G' in props.name and (props.major,props.minor)==(8,6),'A10G identity differs')
 require(torch.cuda.is_bf16_supported() and props.total_memory>=22*1024**3,'BF16 or device memory unsupported')
 write(run/'artifacts/environment.json',{'python':sys.version,'headers':str(headers),'package_pins':actual,'all_installed_packages':sorted(f'{d.metadata["Name"]}=={d.version}' for d in importlib.metadata.distributions()),'device':{'name':props.name,'compute_capability':[props.major,props.minor],'total_memory_bytes':props.total_memory,'BF16_supported':True},'torch_cuda':torch.version.cuda,'model_repository':'openbmb/MiniCPM5-2B-Midtrain','model_revision':REVISION})
 require('HF_TOKEN' not in os.environ,'public setup must not receive private token')
 from huggingface_hub import hf_hub_download
 model=run/'model';model.mkdir()
 for name,digest in MODEL_PINS.items():
  path=hf_hub_download('openbmb/MiniCPM5-2B-Midtrain',filename=name,revision=REVISION,local_dir=model,token=False)
  require(Path(path).resolve()==(model/name).resolve() and sha(path)==digest,'public model hash differs')
 write(run/'artifacts/model-download.json',{'repository':'openbmb/MiniCPM5-2B-Midtrain','revision':REVISION,'files':MODEL_PINS,'anonymous':True,'status':'verified'})
if __name__=='__main__':main()
