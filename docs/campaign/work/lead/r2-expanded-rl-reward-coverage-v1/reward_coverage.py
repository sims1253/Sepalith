#!/usr/bin/env python3
"""Per-row syntax coverage adapter that retains every admitted TRAIN row."""
from __future__ import annotations
from typing import Any,Mapping
class CoverageError(ValueError):pass
def require(ok:bool,why:str):
 if not ok:raise CoverageError(why)
def adapt_evidence(row:Mapping[str,Any])->dict[str,Any]:
 require(row.get('split')=='train' and isinstance(row.get('row_id'),str),'train_identity')
 if row.get('supported'):
  mode=row.get('buffer_mode');require(mode in {'complete_document','completion_prefix','framed_fragment'},'verified_mode')
  return {'row_id':row['row_id'],'syntax_evidence_mode':mode,'syntax_available':True,'repair_reason':None,
          'context_sha256':row['replacement_range']['content_sha256']}
 require(isinstance(row.get('repair_reason'),str) and row['repair_reason'],'repair_reason_missing')
 return {'row_id':row['row_id'],'syntax_evidence_mode':'unverified','syntax_available':False,
         'repair_reason':row['repair_reason'],'context_sha256':row['replacement_range']['content_sha256']}
def score_gate(*,exact:bool,protocol_valid:bool,false_noop_edit:bool,repetition:bool,syntax_status:str)->tuple[float,str]:
 """Prepared reward ordering; syntax unavailable is neutral, never beneficial."""
 require(syntax_status in {'passed','failed','unavailable'},'syntax_status')
 if not protocol_valid:return -1.0,'invalid_or_unterminated'
 if exact:return 1.2,'exact_reference'
 if false_noop_edit:return -1.0,'false_noop_edit'
 if repetition:return -0.75,'severe_repetition'
 if syntax_status=='failed':return -0.5,'applied_syntax_invalid'
 return 0.0,('syntax_unavailable_no_credit' if syntax_status=='unavailable' else 'wrong_parseable_no_semantic_credit')
