"""Rechunk retained token IDs without opening raw corpus or model files."""
import json,random
from collections import Counter
from pathlib import Path
import raw_cpt_v2 as c
HERE=Path(__file__).resolve().parent

def main():
    source=HERE/'profile-shard-v1';out=HERE/'profile-shard-v2-2k';out.mkdir(exist_ok=False)
    old=json.loads((source/'manifest.json').read_text())
    for name,meta in old['artifacts'].items():
        if c.sha(source/name)!=meta['sha256']:raise ValueError('source artifact changed')
    document_rows={r['document_id']:r for r in map(json.loads,(source/'documents.jsonl').open())}
    counts={};all_ids={};doc_ids={}
    for part in ('cpt_train','cpt_validation'):
        stat=Counter();row_ids=[];part_docs=set()
        with (out/(part+'.jsonl')).open('w') as f:
            current=None;tokens=[];first=None
            def flush():
                if current is None:return
                expected=document_rows[current]
                if len(tokens)!=expected['source_code_tokens']:raise ValueError('source token count mismatch')
                for index,chunk in enumerate(c.chunks(tokens,2048)):
                    row={k:first[k] for k in ('schema','document_id','package','group_id','cpt_partition','source_path','source_sha256')}
                    row.update(row_id=current+':'+str(index),chunk_index=index,**chunk)
                    f.write(json.dumps(row,separators=(',',':'))+'\n');row_ids.append(row['row_id'])
                    stat['rows']+=1;stat['input_tokens']+=len(row['input_ids']);stat['supervised_tokens']+=row['supervised_tokens']
                stat['code_tokens']+=len(tokens);stat['documents']+=1;part_docs.add(current)
            for row in map(json.loads,(source/(part+'.jsonl')).open()):
                if row['document_id']!=current:
                    flush();current=row['document_id'];tokens=[];first=row
                tokens.extend(x for x in row['labels'] if x not in (-100,c.EOS))
            flush()
        if stat['supervised_tokens']!=stat['code_tokens']+stat['documents']:raise ValueError('loss denominator mismatch')
        stat['packages']=old['counts'][part]['packages'];stat['raw_bytes']=old['counts'][part]['raw_bytes']
        counts[part]=dict(stat);all_ids[part]=row_ids;doc_ids[part]=part_docs
    if doc_ids['cpt_train'] & doc_ids['cpt_validation']:raise ValueError('cross-partition document duplicate')
    draws=all_ids['cpt_train'].copy();random.Random(3407).shuffle(draws)
    schedule={'schema':1,'seed':3407,'policy':'one deterministic without-replacement permutation; no repeated rows',
              'row_ids':draws,'draw_count':len(draws),'profile_prefix_draws':{'25_steps_effective16':400,'50_steps_effective16':800}}
    (out/'draws-one-pass.json').write_text(json.dumps(schedule,separators=(',',':'))+'\n')
    manifest={**old,'counts':counts,'max_length':2048,'source_profile_manifest_sha256':c.sha(source/'manifest.json'),
              'source_sha256':c.sha(HERE/'raw_cpt_v2.py'),'derivation_source_sha256':c.sha(__file__),
              'boundaries':'Every chunk is BOS+payload+EOS. Continuations include one masked previous source token. Nonterminal EOS labels=-100. Exact document-final EOS is supervised. No code token discarded or supervised twice.',
              'artifacts':{p.name:{'sha256':c.sha(p),'bytes':p.stat().st_size} for p in sorted(out.iterdir())}}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'manifest_sha256':c.sha(out/'manifest.json'),'counts':counts,'artifacts':manifest['artifacts']}))

if __name__=='__main__':main()
