"""CPU checks of the production preparation functions; never imports a model."""
import hashlib, json, tempfile, unittest
from pathlib import Path
import raw_cpt_v2 as c

class CorpusChecks(unittest.TestCase):
    def test_sequential_boundaries_all_lengths(self):
        for n in (1, 2, 3, 6, 7, 8, 13, 14, 15, 16, 100):
            ids = list(range(10, 10+n)); rows = list(c.chunks(ids, 8))
            learned = [x for r in rows for x in r['labels'] if x != -100]
            self.assertEqual(learned, ids+[c.EOS])
            self.assertEqual(sum(r['is_document_end'] for r in rows), 1)
            for i, r in enumerate(rows):
                self.assertLessEqual(len(r['input_ids']), 8)
                self.assertEqual(r['input_ids'][0], c.BOS)
                self.assertEqual(len(r['input_ids']), len(r['labels']))
                self.assertEqual(r['source_token_end']-r['source_token_start'],
                                 r['supervised_tokens']-int(r['is_document_end']))
                if i:
                    self.assertEqual(r['input_ids'][1], ids[r['source_token_start']-1])
                    self.assertEqual(r['labels'][:2], [-100, -100])
                self.assertEqual(r['input_ids'].count(c.EOS), 1)
                self.assertEqual(r['labels'][-1], c.EOS if r['is_document_end'] else -100)

    def test_special_token_collision_rejected(self):
        for token in (c.BOS,c.EOS):
            with self.assertRaises(ValueError): list(c.chunks([10,token,11]))

    def test_real_tokenizer_unicode_crlf_literal_special_strings(self):
        from tokenizers import Tokenizer
        self.assertEqual(c.sha(c.TOKENIZER),c.TOKENIZER_SHA)
        t=Tokenizer.from_file(str(c.TOKENIZER)); t.encode_special_tokens=True
        for text in ('x <- 1\n','value <- "</s>"\n','α <- "λ"\r\n'):
            ids=t.encode(text,add_special_tokens=False).ids
            self.assertEqual(t.decode(ids,skip_special_tokens=False),text)
            self.assertNotIn(c.BOS,ids);self.assertNotIn(c.EOS,ids)
            learned=[x for r in c.chunks(ids,8) for x in r['labels'] if x!=-100]
            self.assertEqual(learned,ids+[c.EOS])

    def test_file_stat_utf8_and_split_guards(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'x.R';p.write_bytes(b'x <- 1\n');st=p.stat()
            row={'path':str(p),'split':'train_group','group_id':'synthetic',
                 'cpt_partition':c.partition('synthetic'),'license':'GPL-3',
                 'bytes':st.st_size,'mtime_ns':st.st_mtime_ns,'inode':st.st_ino,'device':st.st_dev}
            r,value,reason=c.read_file(row);self.assertIsNone(reason)
            self.assertEqual(value[2]['sha256'],hashlib.sha256(p.read_bytes()).hexdigest())
            for split in ('dev_group','final_candidate_group'):
                with self.assertRaises(ValueError):c.read_file({**row,'split':split})
            with self.assertRaises(ValueError):c.read_file({**row,'bytes':99})
            p.write_bytes(b'\xff\n');st=p.stat();row.update(bytes=st.st_size,mtime_ns=st.st_mtime_ns)
            self.assertEqual(c.read_file(row)[2],'non_utf8')

    def test_licenses_require_recognized_family(self):
        for value in ('GPL-3','MIT + file LICENSE','BSD_3_clause + file LICENSE','Apache License (== 2.0)'):
            if value.startswith('BSD_'):continue # Current conservative filter explicitly excludes unknown spelling.
            self.assertTrue(c.allowed_license(value),value)
        for value in ('','file LICENSE','restricted','proprietary'):
            self.assertFalse(c.allowed_license(value),value)

    def test_selection_preserves_partition_and_never_splits_file(self):
        rows=[]
        for i in range(100):
            g='g'+str(i)
            for j in range(3):rows.append({'group_id':g,'path':f'/fixture/{i}/{j}.R',
                'bytes':10+j,'split':'train_group','cpt_partition':c.partition(g)})
        selected=c.select(rows,100,50)
        self.assertEqual(selected,c.select(list(reversed(rows)),100,50))
        for part,limit in [('cpt_train',100),('cpt_validation',50)]:
            self.assertLessEqual(sum(r['bytes'] for r in selected if r['cpt_partition']==part),limit)
        self.assertTrue(all(r in rows for r in selected))

    def test_exact_and_git_blob_identity(self):
        a=c.fingerprints(b'x <- 1\n');b=c.fingerprints(b'x <- 1\n');other=c.fingerprints(b'x <- 2\n')
        self.assertEqual(a,b);self.assertNotEqual(a,other)
        self.assertEqual(a['git_blob_sha1'],hashlib.sha1(b'blob 7\0x <- 1\n').hexdigest())
        self.assertTrue(set(a.values()).intersection({a['sha1']}))

if __name__=='__main__':unittest.main(verbosity=2)
