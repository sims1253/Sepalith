#!/usr/bin/env python3
"""Exact UTF-16/EOL buffer splicing used by expanded TRAIN reward evidence."""
from __future__ import annotations
import hashlib
from typing import Mapping

class BufferError(ValueError): pass

def require(ok: bool, why: str) -> None:
    if not ok: raise BufferError(why)

def sha_text(text: str) -> str:
    return hashlib.sha256(text.encode('utf-8')).hexdigest()

def utf16_index(line: str, units: int) -> int:
    require(type(units) is int and units >= 0, 'invalid_utf16_position')
    used = 0
    for index, char in enumerate(line):
        if used == units: return index
        used += len(char.encode('utf-16-le')) // 2
        require(used <= units, 'utf16_position_splits_code_point')
    require(used == units, 'utf16_position_exceeds_line')
    return len(line)

def position_offset(text: str, line_number: int, units: int) -> int:
    require(type(line_number) is int and line_number >= 0, 'invalid_line_position')
    # splitlines() has no row for an empty string. LSP position (0, 0) is still
    # the valid insertion point in that empty buffer.
    if text == '':
        require(line_number == 0 and units == 0, 'position_exceeds_empty_buffer')
        return 0
    lines = text.splitlines(keepends=True)
    if line_number == len(lines) and text.endswith(('\n', '\r')):
        require(units == 0, 'nonzero_position_after_final_eol')
        return len(text)
    require(line_number < len(lines), 'line_position_exceeds_buffer')
    body = lines[line_number].rstrip('\r\n')
    return sum(len(line) for line in lines[:line_number]) + utf16_index(body, units)

def apply_region(baseline: str, rr: Mapping[str, object], body: str, eol_name: str) -> str:
    require(eol_name in {'lf','crlf'}, 'invalid_document_eol')
    start,end=rr.get('start'),rr.get('end')
    require(isinstance(start,Mapping) and isinstance(end,Mapping), 'replacement_positions_missing')
    left=position_offset(baseline,int(start['line']),int(start['character']))
    right=position_offset(baseline,int(end['line']),int(end['character']))
    require(left <= right, 'replacement_range_reversed')
    eol='\r\n' if eol_name=='crlf' else '\n'
    return baseline[:left]+body.replace('\n',eol)+baseline[right:]
