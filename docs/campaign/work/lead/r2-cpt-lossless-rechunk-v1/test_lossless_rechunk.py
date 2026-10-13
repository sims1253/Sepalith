from __future__ import annotations
import copy, hashlib, importlib.util, json, tempfile, unittest
from pathlib import Path

HERE=Path(__file__).resolve().parent

def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec)
    assert spec.loader is not None;spec.loader.exec_module(m);return m

impl=load('lossless_rechunk_test_impl',HERE/'source/lossless_rechunk.py')
raw=load('lossless_rechunk_test_raw',HERE/'source/raw_cpt_broader.py')
DOC='a'*64
COMMON={'schema':1,'document_id':DOC,'package':'fixture','group_id':'g-trainfixture','cpt_partition':'cpt_train','source_path':'fixture.R','source_sha256':DOC}

def rows_for(tokens,size=7):
    rows=[]
    for i,chunk in enumerate(raw.chunks(tokens,size)):
        rows.append({**COMMON,'row_id':f'{DOC}:{i}','chunk_index':i,**chunk})
    return rows

class LosslessRechunkTests(unittest.TestCase):
    def test_exact_coverage_and_overlap_masks(self):
        tokens=list(range(10,47)); rows=rows_for(tokens)
        common,reassembled,ids=impl.validate_and_reassemble(rows)
        self.assertEqual(tokens,reassembled);self.assertEqual(len(ids),len(rows));self.assertEqual(common['document_id'],DOC)
        for size in (8,16,32):
            out=list(raw.chunks(reassembled,size));seen=[];eos=0
            for index,row in enumerate(out):
                carry=0 if index==0 else 1
                self.assertEqual(row['labels'][:1+carry],[-100]*(1+carry))
                self.assertEqual(row['input_ids'][1+carry:-1],row['labels'][1+carry:-1])
                if carry:self.assertEqual(row['input_ids'][1],seen[-1])
                seen.extend(row['input_ids'][1+carry:-1]);eos+=row['labels'].count(1)
            self.assertEqual(seen,tokens);self.assertEqual(eos,1)

    def assert_rejects(self,rows,pattern):
        with self.assertRaisesRegex(ValueError,pattern): impl.validate_and_reassemble(rows)

    def test_reject_missing_and_reordered_chunks(self):
        rows=rows_for(list(range(10,60)))
        self.assertGreaterEqual(len(rows),3)
        self.assert_rejects([rows[0],*rows[2:]],'missing, reordered')
        swapped=copy.deepcopy(rows);swapped[1],swapped[2]=swapped[2],swapped[1]
        self.assert_rejects(swapped,'missing, reordered')

    def test_reject_truncated_and_mismatched_rows(self):
        rows=rows_for(list(range(10,60)))
        self.assert_rejects(rows[:-1],'truncated')
        bad=copy.deepcopy(rows);bad[1]['package']='other'
        self.assert_rejects(bad,'mismatched document provenance')
        bad=copy.deepcopy(rows);bad[1]['input_ids'][1]+=999
        self.assert_rejects(bad,'prior-token carry mismatch')

    def test_committed_sample_outputs_exact_all_three_contexts(self):
        source=[json.loads(x) for x in (HERE/'sample/input-cpt_train.jsonl').read_text().splitlines()]
        original={}
        for group in impl.documents(iter(json.dumps(x)+'\n' for x in source)):
            common,tokens,_=impl.validate_and_reassemble(group);original[common['document_id']]=tokens
        result=json.loads((HERE/'sample/output/result.json').read_text())
        self.assertEqual(result['totals'],{'documents':3,'input_rows':6,'payload_tokens':8082})
        for size in (8192,16384,32768):
            rows=[json.loads(x) for x in (HERE/f'sample/output/cpt_train_ctx{size}.jsonl').read_text().splitlines()]
            bydoc={}
            for row in rows:
                self.assertEqual(row['row_id'],f"{row['document_id']}:ctx{size}:{row['chunk_index']}")
                bydoc.setdefault(row['document_id'],[]).append(row)
            self.assertEqual(set(bydoc),set(original))
            for ident,parts in bydoc.items():
                rebuilt=[]
                for index,row in enumerate(parts):
                    carry=0 if index==0 else 1
                    self.assertEqual(row['labels'][:1+carry],[-100]*(1+carry))
                    rebuilt.extend(row['input_ids'][1+carry:-1])
                self.assertEqual(rebuilt,original[ident])
                self.assertEqual(sum(p['labels'].count(1) for p in parts),1)

    def test_failed_run_publishes_nothing_and_retry_is_safe(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp=Path(tmp);bad=tmp/'bad.jsonl';bad.write_text(json.dumps(rows_for(list(range(10,30)))[0])+'\n')
            manifest=json.loads((HERE/'sample/input-manifest.json').read_text());item=manifest['inputs'][0]
            item.update(path=str(bad),bytes=bad.stat().st_size,sha256=hashlib.sha256(bad.read_bytes()).hexdigest(),rows=1,documents=1,payload_tokens=20)
            manifest['expected_totals']={'rows':1,'documents':1,'payload_tokens':20}
            mp=tmp/'manifest.json';mp.write_text(json.dumps(manifest))
            out=tmp/'published'
            with self.assertRaisesRegex(ValueError,'truncated'):impl.rechunk(mp,out)
            self.assertFalse(out.exists())
            self.assertEqual(list(tmp.glob('published.tmp-*')),[])
            goodrows=rows_for(list(range(10,30)),7);bad.write_text(''.join(json.dumps(x)+'\n' for x in goodrows))
            item.update(bytes=bad.stat().st_size,sha256=hashlib.sha256(bad.read_bytes()).hexdigest(),rows=len(goodrows))
            manifest['expected_totals']['rows']=len(goodrows);mp.write_text(json.dumps(manifest))
            result=impl.rechunk(mp,out)
            self.assertEqual(result['status'],'complete');self.assertTrue(out.is_dir())

if __name__=='__main__':unittest.main()
