from pathlib import Path
from datetime import datetime,timezone
import ast,hashlib,json,os,subprocess
os.environ['CUDA_VISIBLE_DEVICES']=''
from transformers import AutoTokenizer
base=Path('/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb/docs/campaign')
model=Path('/mnt/e/sepalith/campaign-20260915/models/minicpm5-2b-midtrain')
gguf=Path('/mnt/e/sepalith/campaign-20260915/models/minicpm5-2b-midtrain-vocab-only.gguf')
bin=Path('/home/m0hawk/Documents/Sepalith/experiments/bin/llama/llama-b10453/llama-tokenize')
t=AutoTokenizer.from_pretrained(model,local_files_only=True)
cases={'plain_R':'f <- function(x) {\n  mean(x, na.rm = TRUE)\n}', 'unicode':"café <- 'λ😀'\n  ", 'whitespace':'\t x <- 1  \n\n', 'newline_variants':'x\r\ny\n', 'empty':'','no_op':'[NO_EDIT]\n>>>>>>> UPDATED','deletion':'>>>>>>> UPDATED','quoted_marker':"marker <- '>>>>>>> UPDATED'\n>>>>>>> UPDATED",'literal_eos':"literal <- '</s>'",'literal_bos':"literal <- '<s>'",'literal_endoftext':'<|endoftext|>','prompt_boundary':'=======\n<[fim-middle]>'}
rows=[]
for ident,text in cases.items():
 for bos in (False,True):
  cmd=[str(bin),'-m',str(gguf),'--ids','--stdin','--no-escape','--offline']+([] if bos else ['--no-bos'])
  out=subprocess.run(cmd,input=text,text=True,capture_output=True,timeout=30,check=True)
  native=ast.literal_eval(out.stdout.strip());hf=t.encode(text,add_special_tokens=bos)
  rows.append({'id':ident,'add_bos':bos,'text_sha256':hashlib.sha256(text.encode()).hexdigest(),'hf_ids':hf,'llama_ids':native,'equal':hf==native,'source_special_token_present':any(v in (0,1) for v in t.encode(text,add_special_tokens=False))})
seams=[]
for prompt in ('=======\n<[fim-middle]>','=======\n<[fim-middle]>\n'):
 for target in ('[NO_EDIT]\n>>>>>>> UPDATED','x\n>>>>>>> UPDATED','\n  x  \n>>>>>>> UPDATED','>>>>>>> UPDATED'):
  p=t.encode(prompt,add_special_tokens=False);joined=t.encode(prompt+target,add_special_tokens=False)
  seams.append({'prompt_trailing_lf':prompt.endswith('\n'),'target':target,'prompt_is_token_prefix_of_joint':joined[:len(p)]==p,'separate_equals_joint':p+t.encode(target,add_special_tokens=False)==joined})
r={'task':'PRM-04','observed_at':datetime.now(timezone.utc).isoformat(),'method':'Pinned CPU llama-tokenize using metadata-only GGUF; default parse_special=true; no server, forward pass or CUDA','tokenizer_revision':'8dc5f6055b90fe4b9422340810b270b9569f37f3','tokenizer_sha256':hashlib.sha256((model/'tokenizer.json').read_bytes()).hexdigest(),'gguf':str(gguf),'gguf_sha256':hashlib.sha256(gguf.read_bytes()).hexdigest(),'gguf_bytes':gguf.stat().st_size,'binary':str(bin),'binary_sha256':hashlib.sha256(bin.read_bytes()).hexdigest(),'cases':rows,'matches':sum(x['equal'] for x in rows),'comparisons':len(rows),'seam_checks':seams,'limitation':'These are tokenizer/protocol probes, not new renderer shared fixtures or serving/generation tests.'}
(base/'receipts/PRM-04-lead-token-parity.json').write_text(json.dumps(r,indent=2)+'\n')
print(json.dumps({'matches':r['matches'],'comparisons':r['comparisons'],'gguf_sha256':r['gguf_sha256'],'mismatches':[x for x in rows if not x['equal']],'seams':seams},indent=2))
