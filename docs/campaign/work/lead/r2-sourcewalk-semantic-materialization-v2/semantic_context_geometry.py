#!/usr/bin/env python3
"""Shared, side-effect-free geometry for semantic TRAIN/serving contexts."""
from __future__ import annotations
import hashlib
from typing import Any,Mapping,Sequence

def sha_bytes(value:bytes)->str:return hashlib.sha256(value).hexdigest()
def eol_text(mode:str)->str:
 if mode=='lf':return '\n'
 if mode=='crlf':return '\r\n'
 raise ValueError('unsupported_document_eol')
def context_buffer_text(context:Mapping[str,Any])->str:
 """Reconstruct the exact editable buffer represented by PromptContext."""
 prefix=context.get('prefix');old=context.get('region_old');suffix=context.get('suffix_lines')
 if not all(isinstance(x,list) and all(isinstance(y,str) for y in x) for x in (prefix,old,suffix)):raise ValueError('context_line_arrays_invalid')
 rr=context.get('replacement_range',{});start=rr.get('start');end=rr.get('end')
 if not isinstance(start,dict) or not isinstance(end,dict):raise ValueError('replacement_positions_missing')
 if old:lines=[*prefix,*old,*suffix]
 else:lines=[*prefix,'',*suffix]
 text=eol_text(context.get('document_eol')).join(lines)
 line=len(prefix)
 if start!={'line':line,'character':0} or end!={'line':line,'character':0} or old:raise ValueError('expected_zero_width_blank_anchor')
 if lines[line] != '':raise ValueError('physical_blank_anchor_missing')
 if sha_bytes(text.encode('utf-8'))!=rr.get('content_sha256'):raise ValueError('context_content_sha256_mismatch')
 return text

def apply_zero_width(context:Mapping[str,Any],body_lines:Sequence[str])->str:
 """Apply an insertion exactly as the inline editor path at a blank line."""
 if not body_lines or any(not isinstance(x,str) or '\n' in x or '\r' in x for x in body_lines):raise ValueError('body_lines_invalid')
 text=context_buffer_text(context);separator=eol_text(context['document_eol']);line=context['replacement_range']['start']['line'];lines=text.split(separator)
 if lines[line] != '':raise ValueError('physical_blank_anchor_missing')
 lines[line]=separator.join(body_lines)
 return separator.join(lines)

def normalize_source(raw:bytes)->tuple[str,list[str],str]:
 text=raw.decode('utf-8')
 if '\r' in text.replace('\r\n',''):raise ValueError('mixed_or_lone_cr_source')
 mode='crlf' if '\r\n' in text else 'lf';separator=eol_text(mode);return text,text.split(separator),mode

def derive_full_before(source_after:bytes,target_lines:Sequence[str],target_function_span:Sequence[int])->dict[str,Any]:
 """Invert one target block to a physical blank anchor and verify reapply."""
 text,lines,mode=normalize_source(source_after);separator=eol_text(mode);target=separator.join(target_lines)
 occurrences=sum(1 for i in range(len(lines)-len(target_lines)+1) if lines[i:i+len(target_lines)]==list(target_lines))
 if occurrences!=1:raise ValueError('target_line_block_not_unique')
 start=next(i for i in range(len(lines)-len(target_lines)+1) if lines[i:i+len(target_lines)]==list(target_lines));end=start+len(target_lines)
 if not isinstance(target_function_span,(list,tuple)) or len(target_function_span)!=2 or any(type(x) is not int for x in target_function_span):raise ValueError('target_function_span_invalid')
 function_start=target_function_span[0]-1
 if function_start<end or any(lines[i] != '' for i in range(end,function_start)):raise ValueError('target_function_not_after_only_blank_gap')
 # Replace only the target.  Keep every intervening blank and every source
 # line after it.  The new empty line is the physical zero-width insertion
 # anchor; applying the target to that line reconstructs source_after exactly.
 before_lines=[*lines[:start],'',*lines[end:]];before=separator.join(before_lines)
 context={'prefix':before_lines[:start],'region_old':[],'suffix_lines':before_lines[start+1:],'document_eol':mode,'replacement_range':{'start':{'line':start,'character':0},'end':{'line':start,'character':0},'content_sha256':sha_bytes(before.encode())}}
 if apply_zero_width(context,target_lines)!=text:raise ValueError('source_inverse_apply_mismatch')
 return {'before_text':before,'before_sha256':sha_bytes(before.encode()),'before_lines':before_lines,'target_start_line':start,'document_eol':mode,'reapplied_after_sha256':sha_bytes(text.encode()),'geometry':context}
