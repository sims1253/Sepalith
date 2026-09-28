"""Minimal reviewed generation-state capture used by the DEV-only process."""
from copy import deepcopy
RUNTIME_CONFIG_FIELDS=('use_cache','bos_token_id','eos_token_id','pad_token_id')
RUNTIME_TOKENIZER_FIELDS=('padding_side','pad_token','pad_token_id','bos_token_id','eos_token_id')
def clear_generation_markers(model):
 count=0
 for module in model.modules():
  namespace=getattr(module,'__dict__',None)
  if namespace is not None and namespace.pop('_flag_for_generation',None) is not None:count+=1
 return count
def capture_runtime_contract(model):
 configs={};tokenizers={}
 for module in model.modules():
  for attr in ('config','generation_config'):
   config=getattr(module,attr,None)
   if config is not None:
    values={field:deepcopy(getattr(config,field)) for field in RUNTIME_CONFIG_FIELDS if hasattr(config,field)}
    if values:configs[id(config)]=(config,values)
  tokenizer=getattr(module,'_saved_temp_tokenizer',None)
  if tokenizer is not None:
   values={field:deepcopy(getattr(tokenizer,field)) for field in RUNTIME_TOKENIZER_FIELDS if hasattr(tokenizer,field)}
   if values:tokenizers[id(tokenizer)]=(tokenizer,values)
 return configs,tokenizers
def restore_runtime_contract(configs,tokenizers):
 for config,values in configs.values():
  for field,value in values.items():setattr(config,field,value)
 for tokenizer,values in tokenizers.values():
  for field,value in values.items():setattr(tokenizer,field,value)
