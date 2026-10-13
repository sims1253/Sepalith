#!/usr/bin/env python3
from __future__ import annotations
import json,sys,tempfile,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from buffer_geometry import apply_region,position_offset,sha_text
from materialize_expanded_buffers import SourceCache,reconstruct

OLD=Path('/mnt/e/sepalith/campaign-20260915/data-work/RL11-expanded-context-v1/context-sidecar.jsonl')
NEW=Path('/mnt/e/sepalith/campaign-20260915/data-work/DAT10-expanded-15008-v1/candidate-context-provenance.jsonl')

class BufferPreparationTest(unittest.TestCase):
 def test_empty_buffer_position_and_insertion(self):
  self.assertEqual(position_offset('',0,0),0)
  rr={'start':{'line':0,'character':0},'end':{'line':0,'character':0},'content_sha256':sha_text('')}
  self.assertEqual(apply_region('',rr,'x <- 1\n','lf'),'x <- 1\n')
 def test_empty_buffer_rejects_other_positions(self):
  for line,char in [(1,0),(0,1)]:
   with self.subTest(line=line,char=char),self.assertRaises(ValueError):position_offset('',line,char)
 def test_snapshot_history_reconstructs_exact_context(self):
  with OLD.open() as handle: row=json.loads(handle.readline())
  text,mode,suffix,_=reconstruct(row,'rl_context_sidecar',SourceCache())
  self.assertEqual(sha_text(text),row['context']['replacement_range']['content_sha256']);self.assertEqual(mode,'complete_document')
  self.assertEqual(suffix,'')
 def test_new_finish_builder_is_provenance_framed_fragment(self):
  with NEW.open() as handle: row=json.loads(handle.readline())
  text,mode,suffix,_=reconstruct(row,'candidate_context_provenance',SourceCache())
  self.assertEqual(sha_text(text),row['context']['replacement_range']['content_sha256']);self.assertEqual(mode,'framed_fragment')
  self.assertEqual(suffix,'\n}')
 def test_long_target_splice_is_not_truncated(self):
  baseline='f <- function() {\n}\n';rr={'start':{'line':1,'character':0},'end':{'line':1,'character':0}}
  body='\n'.join(f'  x{i} <- {i}' for i in range(400))
  applied=apply_region(baseline,rr,body,'lf');self.assertIn('x399 <- 399',applied);self.assertEqual(applied.count('\n'),401)

if __name__=='__main__':unittest.main()
