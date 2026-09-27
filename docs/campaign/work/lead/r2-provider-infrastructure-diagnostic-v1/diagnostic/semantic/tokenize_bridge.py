#!/usr/bin/env python3
import json,sys
from tokenizers import Tokenizer
path=sys.argv[1];tokenizer=Tokenizer.from_file(path);tokenizer.encode_special_tokens=True
for line in sys.stdin:
 try:
  request=json.loads(line);ids=tokenizer.encode(request['text'],add_special_tokens=False).ids
  print(json.dumps({'id':request['id'],'ids':ids},separators=(',',':')),flush=True)
 except Exception as e:
  print(json.dumps({'error':type(e).__name__+':'+str(e)},separators=(',',':')),flush=True)
