"""Narrow adapter from an admitted CPT recipe to the streaming dataset."""
import hashlib,json,sqlite3
from pathlib import Path
from campaign_cpt_data import read_jsonl
from cpt_streaming_cache import StreamingCptDataset,require,sha256

class StageCursorView:
 """Dataset-managed cursor used when Trainer data skipping is intentionally off."""
 def __init__(self,base,cursor):
  require(type(cursor)is int and 0<=cursor<=len(base),'stage cursor outside schedule');self.base=base;self.initial_cursor=cursor
 def __len__(self):return len(self.base)-self.initial_cursor
 def __getitem__(self,position):
  require(type(position)is int and 0<=position<len(self),'stage-local position outside remaining schedule');return self.base[self.initial_cursor+position]
 def close(self):self.base.close()

def dataset_from_bound_recipe(recipe,initial_cursor):
 cohort=recipe['cohort'];binding=cohort.get('streaming_cache');require(isinstance(binding,dict),'streaming cache remains unbound');root=Path(binding['path']);manifest_path=root/'manifest.json';require(manifest_path.is_file() and sha256(manifest_path)==binding['manifest_sha256'],'streaming manifest differs')
 manifest=json.loads(manifest_path.read_text());require(manifest['source']['rows']['sha256']==cohort['rows']['sha256'],'cache row identity differs');require(manifest['source']['draw_schedule']['sha256']==cohort['draw_schedule']['sha256'],'cache schedule identity differs')
 expected={'rows':cohort['unique_rows'],'documents':cohort['documents'],'packages':cohort['packages'],'input_tokens':cohort['input_tokens'],'payload_tokens':cohort['payload_tokens'],'loss_tokens':cohort['loss_tokens'],'draws':cohort['updates']*16,'named_replays':cohort['named_replays'],'updates':cohort['updates']}
 require(all(manifest['counts'][k]==v for k,v in expected.items()),'streaming cache denominators differ');require(manifest['max_sequence_tokens']==cohort['max_sequence_tokens'],'streaming context differs')
 validation=read_jsonl(recipe['validation']['path'],max_sequence_tokens=recipe['validation']['max_sequence_tokens'],materialized=True);require(len(validation)==recipe['validation']['rows']==499,'validation denominator differs');valid_packages={r['package'] for r in validation};valid_documents={r['source_sha256'] for r in validation}
 db=sqlite3.connect(f'file:{root/"index.sqlite3"}?mode=ro',uri=True)
 try:
  for package,document_sha in db.execute('select package,source_sha256 from documents'):
   require(package not in valid_packages and document_sha not in valid_documents,'CPT holdout leakage')
 finally:db.close()
 base=StreamingCptDataset(root,binding['manifest_sha256'],initial_cursor)
 return StageCursorView(base,initial_cursor),{'status':'pass','cache_manifest_sha256':binding['manifest_sha256'],'cohort_rows':expected['rows'],'draws':expected['draws'],'remaining_draws':expected['draws']-initial_cursor,'validation_rows':499,'initial_cursor':initial_cursor,'cursor_owner':'dataset_view_ignore_data_skip_true','cuda_started':False}
