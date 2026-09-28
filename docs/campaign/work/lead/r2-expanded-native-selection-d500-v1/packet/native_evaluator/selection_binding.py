"""Root-bound R2 model selection; never infer a model from a legacy path."""
import re
from pathlib import Path
TOK='3e065a558a034185fe299917b398685c1facd0169a9eea1e629eb30c171fed81'
TOKCFG='e9b1064649e771d7a8e15637c68b2d2749724877ba3648bd74a4ece31c26303b'
def validate_profile(p):
 s=p.get('selection',{});m=p['model'];mp=p['model_profile']
 for value in [m.get('sha256'),s.get('merged_parent_manifest_sha256'),s.get('export_integrity_receipt_sha256'),s.get('task_recipe_sha256'),s.get('checkpoint_manifest_sha256')]:
  if not isinstance(value,str) or re.fullmatch('[0-9a-f]{64}',value) is None or value in ('0'*64,'f'*64):raise ValueError('root R2 selection is unbound')
 if s.get('task_source')!='bb2f9fdc1d016679533beab7d9906c9b996c9ce59d36e54ff98101adbaeed384' or s.get('checkpoint_step')!=500:raise ValueError('expanded checkpoint/source differs')
 if s.get('purpose')!='development_selection_only' or s.get('parent_kind')!='sft_merged':raise ValueError('expanded parent/DEV role required')
 for name in ('q8_path','tokenizer_dir'):
  q=Path(s[name])
  if not q.is_absolute() or '..' in q.parts:raise ValueError('selected artifact path differs')
 if Path(s['q8_path']).name!='model-Q8_0.gguf' or type(m.get('bytes')) is not int or m['bytes']<=1_000_000_000:raise ValueError('Q8 artifact identity missing')
 if mp['modelSha256']!=m['sha256'] or mp['modelRevision']!=s['revision']:raise ValueError('model profile identity differs')
 if mp['tokenizerJsonSha256']!=TOK or mp['tokenizerConfigSha256']!=TOKCFG or mp['renderer']!='zeta2-prm03-v1':raise ValueError('PRM03/tokenizer contract differs')
 if (p['backend'],p['build'],p['cuda_graph_opt'],p['batch'],p['ubatch'],p['context'],p['output'],p['parallel'])!=('cuda','b10453',0,256,256,4096,192,1):raise ValueError('delivered native profile differs')
 return s
