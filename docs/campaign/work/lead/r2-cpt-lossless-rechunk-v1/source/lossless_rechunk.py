#!/usr/bin/env python3
"""Losslessly re-chunk frozen 2048-token CPT rows without source reads/tokenization."""
from __future__ import annotations
import argparse, hashlib, importlib.util, json, os, shutil, sys, uuid
from pathlib import Path

BOS, EOS, MASK = 0, 1, -100
REQUIRED_COMMON = ('document_id','package','group_id','cpt_partition','source_path','source_sha256','document_token_count')


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False)


def sha256(path: Path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()


def load_chunks(path: Path, expected_sha: str):
    if sha256(path) != expected_sha: raise ValueError('raw chunks source SHA-256 mismatch')
    spec=importlib.util.spec_from_file_location('sepalith_pinned_raw_chunks',path)
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module
    assert spec.loader is not None;spec.loader.exec_module(module)
    return module.chunks


def validate_and_reassemble(rows):
    if not rows: raise ValueError('empty document row group')
    common={key:rows[0].get(key) for key in REQUIRED_COMMON}
    ident=common['document_id']
    if not isinstance(ident,str) or len(ident)!=64 or common['source_sha256']!=ident:
        raise ValueError('document/source identity mismatch')
    if common['cpt_partition']!='cpt_train': raise ValueError(f'{ident}: non-TRAIN partition')
    count=common['document_token_count']
    if not isinstance(count,int) or count<=0: raise ValueError(f'{ident}: invalid document token count')
    expected_start=0; payload=[]; original_ids=[]
    for index,row in enumerate(rows):
        if any(row.get(k)!=v for k,v in common.items()): raise ValueError(f'{ident}: mismatched document provenance')
        if row.get('schema')!=1 or row.get('chunk_index')!=index or row.get('row_id')!=f'{ident}:{index}':
            raise ValueError(f'{ident}: missing, reordered, or mismatched chunk index/row ID')
        start=row.get('source_token_start');end=row.get('source_token_end')
        if row.get('token_start')!=start or row.get('token_end')!=end or start!=expected_start:
            raise ValueError(f'{ident}: non-contiguous source range')
        if not isinstance(end,int) or end<=start or end>count: raise ValueError(f'{ident}: invalid source range')
        carry=0 if index==0 else 1
        if row.get('overlap_context_tokens')!=carry: raise ValueError(f'{ident}: overlap count mismatch')
        inputs=row.get('input_ids');labels=row.get('labels');attention=row.get('attention_mask')
        if not all(isinstance(x,list) for x in (inputs,labels,attention)) or not (len(inputs)==len(labels)==len(attention)):
            raise ValueError(f'{ident}: tensor vector lengths mismatch')
        if inputs[0]!=BOS or inputs[-1]!=EOS or attention!=[1]*len(inputs):
            raise ValueError(f'{ident}: BOS/EOS/attention contract mismatch')
        body=inputs[1+carry:-1]
        if len(body)!=end-start or labels[:1+carry]!=[MASK]*(1+carry) or labels[1+carry:-1]!=body:
            raise ValueError(f'{ident}: payload labels/masks mismatch')
        if any(token in (BOS,EOS) for token in body): raise ValueError(f'{ident}: special token in source payload')
        if carry and (inputs[1]!=payload[-1] or labels[1]!=MASK): raise ValueError(f'{ident}: prior-token carry mismatch')
        terminal=end==count
        if row.get('is_document_end') is not terminal or labels[-1]!=(EOS if terminal else MASK):
            raise ValueError(f'{ident}: terminal EOS contract mismatch')
        supervised=sum(x!=MASK for x in labels)
        if row.get('supervised_tokens')!=supervised: raise ValueError(f'{ident}: supervised token count mismatch')
        payload.extend(body);expected_start=end;original_ids.append(row['row_id'])
    if expected_start!=count or len(payload)!=count: raise ValueError(f'{ident}: truncated document')
    if not rows[-1]['is_document_end']: raise ValueError(f'{ident}: missing terminal source chunk')
    return common,payload,original_ids


def documents(stream):
    current_id=None;group=[];closed=set()
    for line_number,line in enumerate(stream,1):
        try: row=json.loads(line)
        except Exception as error: raise ValueError(f'invalid JSON at line {line_number}') from error
        ident=row.get('document_id')
        if current_id is None: current_id=ident
        if ident!=current_id:
            closed.add(current_id);yield group
            if ident in closed: raise ValueError(f'non-contiguous repeated document: {ident}')
            current_id=ident;group=[]
        group.append(row)
    if group: yield group


def rechunk(manifest_path: Path, output: Path):
    manifest=json.loads(manifest_path.read_text())
    if manifest.get('schema')!='sepalith.cpt.lossless-rechunk-input.v1': raise ValueError('input manifest schema mismatch')
    sizes=manifest.get('context_sizes')
    if sizes!=[8192,16384,32768]: raise ValueError('context sizes must be exactly 8192/16384/32768')
    raw=manifest['raw_chunks'];chunker=load_chunks(Path(raw['path']),raw['sha256'])
    if (raw.get('bos'),raw.get('eos'),raw.get('source_chunk_size'))!=(BOS,EOS,2048): raise ValueError('raw chunks interface binding mismatch')
    if output.exists(): raise FileExistsError(f'fresh output required: {output}')
    stage=output.with_name(output.name+'.tmp-'+uuid.uuid4().hex);stage.mkdir(parents=True)
    streams={size:(stage/f'cpt_train_ctx{size}.jsonl').open('x') for size in sizes}
    provenance=(stage/'document-provenance.jsonl').open('x')
    totals={'input_rows':0,'documents':0,'payload_tokens':0}
    output_stats={str(size):{'rows':0,'input_tokens':0,'supervised_tokens':0,'payload_tokens':0,'terminal_eos':0} for size in sizes}
    seen=set(); input_artifacts=[]
    try:
        for item in manifest['inputs']:
            path=Path(item['path'])
            if path.stat().st_size!=item['bytes'] or sha256(path)!=item['sha256']: raise ValueError(f'input artifact differs: {path}')
            local={'rows':0,'documents':0,'payload_tokens':0}
            with path.open() as source:
                for group in documents(source):
                    common,payload,original_ids=validate_and_reassemble(group);ident=common['document_id']
                    if ident in seen: raise ValueError(f'duplicate document across inputs: {ident}')
                    seen.add(ident); local['rows']+=len(group);local['documents']+=1;local['payload_tokens']+=len(payload)
                    provenance.write(canonical({
                        'schema':'sepalith.cpt.lossless-rechunk-provenance.v1',**common,
                        'original_row_ids':original_ids,'original_chunk_count':len(group),
                        'original_context_size':2048,'input_artifact_sha256':item['sha256'],
                    })+'\n')
                    for size in sizes:
                        produced=list(chunker(payload,size))
                        for index,chunk in enumerate(produced):
                            record={
                                'schema':2,'row_id':f'{ident}:ctx{size}:{index}','document_id':ident,
                                'package':common['package'],'group_id':common['group_id'],'cpt_partition':'cpt_train',
                                'source_path':common['source_path'],'source_sha256':common['source_sha256'],
                                'chunk_index':index,'context_size':size,'parent_document_id':ident,
                                'original_context_size':2048,'original_chunk_count':len(group),
                                'original_first_row_id':original_ids[0],'original_last_row_id':original_ids[-1],
                                'input_artifact_sha256':item['sha256'],**chunk,
                            }
                            streams[size].write(canonical(record)+'\n')
                            stats=output_stats[str(size)];stats['rows']+=1;stats['input_tokens']+=len(chunk['input_ids']);stats['supervised_tokens']+=chunk['supervised_tokens'];stats['payload_tokens']+=chunk['source_token_end']-chunk['source_token_start'];stats['terminal_eos']+=int(chunk['is_document_end'])
                    totals['documents']+=1;totals['payload_tokens']+=len(payload);totals['input_rows']+=len(group)
            expected={'rows':item['rows'],'documents':item['documents'],'payload_tokens':item['payload_tokens']}
            if local!=expected: raise ValueError(f'input expected totals mismatch: {path}: {local} != {expected}')
            input_artifacts.append({**item,'verified':True})
        expected=manifest['expected_totals']
        if totals!={'input_rows':expected['rows'],'documents':expected['documents'],'payload_tokens':expected['payload_tokens']}:
            raise ValueError(f'aggregate expected totals mismatch: {totals}')
        for size in sizes:
            stats=output_stats[str(size)]
            if stats['payload_tokens']!=totals['payload_tokens'] or stats['terminal_eos']!=totals['documents']:
                raise ValueError(f'output coverage mismatch for context {size}')
        for stream in [*streams.values(),provenance]: stream.flush();os.fsync(stream.fileno());stream.close()
        artifact_paths=[*sorted(stage.glob('cpt_train_ctx*.jsonl')),stage/'document-provenance.jsonl']
        result={
            'schema':'sepalith.cpt.lossless-rechunk-result.v1','status':'complete','manifest_path':str(manifest_path),
            'manifest_sha256':sha256(manifest_path),'raw_chunks_sha256':raw['sha256'],'inputs':input_artifacts,
            'totals':totals,'outputs':output_stats,
            'artifacts':{p.name:{'bytes':p.stat().st_size,'sha256':sha256(p)} for p in artifact_paths},
            'guarantees':{'retokenized':False,'raw_source_read':False,'all_source_tokens_once_per_context':True,'single_terminal_eos_per_document':True,'prior_token_overlap_masked':True},
        }
        rp=stage/'result.json';rp.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
        with rp.open('rb') as f: os.fsync(f.fileno())
        dfd=os.open(stage,os.O_RDONLY|os.O_DIRECTORY);os.fsync(dfd);os.close(dfd)
        stage.rename(output)
        return result
    except BaseException:
        for stream in [*streams.values(),provenance]:
            if not stream.closed: stream.close()
        shutil.rmtree(stage)
        raise


def main():
    p=argparse.ArgumentParser();p.add_argument('--input-manifest',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();result=rechunk(args.input_manifest,args.output);print(canonical(result))

if __name__=='__main__': main()
