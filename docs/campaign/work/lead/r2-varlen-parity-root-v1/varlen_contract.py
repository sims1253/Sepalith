#!/usr/bin/env python3
"""Exact CPU contract for one packed 16-row optimizer update."""
from __future__ import annotations
from typing import Any, Mapping, Sequence

class ContractError(ValueError): pass

def pack_update(rows: Sequence[Mapping[str, Any]], *, first_position: int,
                token_cap: int = 16_384, effective_batch: int = 16):
    if len(rows) != effective_batch: raise ContractError("optimizer update membership differs")
    groups=[]; total=0
    for index,row in enumerate(rows):
        ids,labels=row.get('input_ids'),row.get('labels')
        if row.get('_draw_position') != first_position+index: raise ContractError('draw order differs')
        if not isinstance(ids,list) or not isinstance(labels,list) or len(ids)!=len(labels) or not ids: raise ContractError('token/label arrays differ')
        if len(ids)>token_cap: raise ContractError('complete row exceeds physical token cap')
        if labels[0] != -100: raise ContractError('member boundary label is exposed')
        # Only join consecutive rows. First-fit across an earlier group would
        # change the physical forward order inside the fixed 16-row update.
        if not groups or total + len(ids) > token_cap:
            groups.append([]); total = 0
        groups[-1].append(row); total += len(ids)
    return [collate(group) for group in groups]

def collate(rows):
    ids=[];labels=[];positions=[];lengths=[];members=[];cursor=0
    for row in rows:
        length=len(row['input_ids']);rid=row.get('row_id',row.get('id'))
        if not isinstance(rid,str) or not rid:raise ContractError('row ID invalid')
        ids.extend(row['input_ids']);labels.extend(row['labels']);positions.extend(range(length));lengths.append(length)
        members.append({'row_id':rid,'draw_position':row['_draw_position'],'start':cursor,'stop':cursor+length});cursor+=length
    # No attention_mask: Unsloth routes packed_seq_lengths to block-diagonal
    # xFormers or FlashAttention-varlen. Explicit lengths are preferred over
    # inference from reset position_ids.
    return {'input_ids':[ids],'labels':[labels],'position_ids':[positions],
            'packed_seq_lengths':lengths,'members':members,
            'loss_denominator':sum(sum(v!=-100 for v in row['labels'][1:]) for row in rows)}

class PackedUpdateView:
    """Deterministic logical-update view over an existing sequential dataset."""
    def __init__(self, dataset, *, initial_cursor=0, effective_batch=16, token_cap=16_384):
        if type(initial_cursor) is not int or initial_cursor < 0 or initial_cursor % effective_batch:
            raise ContractError('resume cursor is not an optimizer boundary')
        if (len(dataset)-initial_cursor) % effective_batch:
            raise ContractError('remaining schedule is not whole optimizer updates')
        self.dataset=dataset;self.initial_cursor=initial_cursor
        self.effective_batch=effective_batch;self.token_cap=token_cap
    def __len__(self): return (len(self.dataset)-self.initial_cursor)//self.effective_batch
    def __getitem__(self,index):
        if type(index) is not int or not 0<=index<len(self):raise IndexError(index)
        start=self.initial_cursor+index*self.effective_batch
        rows=[self.dataset[i] for i in range(start,start+self.effective_batch)]
        return pack_update(rows,first_position=start,token_cap=self.token_cap,
                           effective_batch=self.effective_batch)

def verify(rows,groups,*,first_position):
    expected={row['_draw_position']:row for row in rows};seen=[];den=0
    for group in groups:
        if 'attention_mask' in group:raise ContractError('packed group must not carry attention_mask')
        flat_ids,flat_labels,pos=group['input_ids'][0],group['labels'][0],group['position_ids'][0]
        if sum(group['packed_seq_lengths']) != len(flat_ids) or not (
            len(flat_ids) == len(flat_labels) == len(pos)
        ):
            raise ContractError('packed geometry differs')
        for member,length in zip(group['members'],group['packed_seq_lengths']):
            row=expected.get(member['draw_position']);a,b=member['start'],member['stop']
            if row is None or b-a!=length or flat_ids[a:b]!=row['input_ids'] or flat_labels[a:b]!=row['labels']:raise ContractError('member bytes changed')
            if pos[a:b]!=list(range(length)) or flat_labels[a]!=-100:raise ContractError('member boundary geometry differs')
            seen.append(member['draw_position'])
        den+=group['loss_denominator']
    if sorted(seen)!=list(range(first_position,first_position+len(rows))):raise ContractError('update membership changed')
    expected_den=sum(sum(v!=-100 for v in row['labels'][1:]) for row in rows)
    if den!=expected_den:raise ContractError('loss denominator changed')
    return {'rows':len(rows),'physical_groups':len(groups),'loss_denominator':den,'tokens':sum(len(r['input_ids']) for r in rows)}
