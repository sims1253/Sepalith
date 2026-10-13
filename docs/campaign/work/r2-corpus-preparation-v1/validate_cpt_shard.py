"""Independently bind stored causal-LM rows back to exact source byte hashes."""
import argparse,hashlib,json,os,time
from collections import Counter
from pathlib import Path
import raw_cpt_v2 as c

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--shard',type=Path,required=True)
    ap.add_argument('--documents',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    if hasattr(os,'sched_setaffinity'):os.sched_setaffinity(0,set(sorted(os.sched_getaffinity(0))[:2]))
    os.environ['TOKENIZERS_PARALLELISM']='false'
    from tokenizers import Tokenizer
    assert c.sha(c.TOKENIZER)==c.TOKENIZER_SHA
    tok=Tokenizer.from_file(str(c.TOKENIZER));tok.encode_special_tokens=True
    declared=json.loads((args.shard/'manifest.json').read_text())
    for name,pin in declared['artifacts'].items():assert c.sha(args.shard/name)==pin['sha256']
    docs={r['document_id']:r for r in map(json.loads,args.documents.open())}
    metadata=json.loads((c.HERE/'raw-train-package-candidates.json').read_text())
    groups={r['name']:r['group_id'] for r in metadata['packages']}
    counts={};seen_docs=set();all_ids=set();started=time.monotonic()
    for part in ('cpt_train','cpt_validation'):
        stat=Counter();current=None;ids=[];first=None;previous=None
        def flush():
            if current is None:return
            assert current not in seen_docs;seen_docs.add(current)
            raw=tok.decode(ids,skip_special_tokens=False).encode('utf8')
            assert hashlib.sha256(raw).hexdigest()==current==first['source_sha256']
            assert len(ids)==first['document_token_count']==docs[current]['source_code_tokens']
            assert len(raw)==docs[current]['source_utf8_bytes']
            assert previous['is_document_end']
            stat['documents']+=1;stat['code_tokens']+=len(ids);stat['source_bytes']+=len(raw)
        for row in map(json.loads,(args.shard/(part+'.jsonl')).open()):
            assert row['row_id'] not in all_ids;all_ids.add(row['row_id'])
            assert row['cpt_partition']==part==c.partition(row['group_id'])
            assert groups[row['package']]==row['group_id']
            rel=Path(row['source_path']).relative_to('/mnt/h/sepalith/normalized')
            assert rel.parts[0]==rel.parts[2]==row['package'] and rel.parts[3]=='R'
            if row['document_id']!=current:
                flush();current=row['document_id'];ids=[];first=row;previous=None
            x,y,attention=row['input_ids'],row['labels'],row['attention_mask']
            assert 3<=len(x)<=declared['max_length']
            assert len(x)==len(y)==len(attention) and attention==[1]*len(x)
            assert x[0]==c.BOS and x[-1]==c.EOS and y[0]==-100
            assert row['source_token_start']==row['token_start']==len(ids)
            assert row['source_token_end']==row['token_end']
            overlap=row['overlap_context_tokens']
            assert overlap==(1 if previous else 0)
            if previous:
                assert x[1]==ids[-1] and y[1]==-100
                assert not previous['is_document_end']
            assert x[1+overlap:-1]==y[1+overlap:-1]
            assert all(t not in (c.BOS,c.EOS) for t in x[1:-1])
            ids.extend(x[1+overlap:-1])
            assert len(ids)==row['token_end']
            terminal=len(ids)==row['document_token_count']
            assert terminal==row['is_document_end']
            assert y[-1]==(c.EOS if terminal else -100)
            stat['supervised_tokens']+=sum(t!=-100 for t in y)
            stat['rows']+=1;stat['input_tokens']+=len(x);previous=row
        flush()
        assert stat['supervised_tokens']==stat['code_tokens']+stat['documents']
        for key in ('supervised_tokens','code_tokens','documents','rows','input_tokens'):
            assert stat[key]==declared['counts'].get(part,{}).get(key,0)
        counts[part]=dict(stat)
    result={'status':'PASS','scope':'All retained row IDs, registry group/path mapping, exact loss labels, sequential offsets, document byte SHA256 reconstructed through pinned tokenizer, cross-partition document dedup; no model or R execution.',
            'counts':counts,'seconds':time.monotonic()-started,'manifest_sha256':c.sha(args.shard/'manifest.json'),
            'validator_sha256':c.sha(__file__),'tokenizer_sha256':c.TOKENIZER_SHA}
    args.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))

if __name__=='__main__':main()
