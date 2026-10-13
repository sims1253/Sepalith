"""Real validator must reject corrupted retained-token fixtures."""
import copy,json,os,subprocess,sys,tempfile,unittest
from collections import defaultdict
from pathlib import Path
import raw_cpt_v2 as c
HERE=Path(__file__).resolve().parent

class ValidatorChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        grouped=defaultdict(list)
        for row in map(json.loads,(HERE/'profile-shard-v2-2k/cpt_train.jsonl').open()):
            grouped[row['document_id']].append(row)
            if row['is_document_end'] and len(grouped[row['document_id']])>=2:
                cls.rows=grouped[row['document_id']];break

    def run_fixture(self, mutate=None):
        rows=copy.deepcopy(self.rows)
        if mutate:mutate(rows)
        with tempfile.TemporaryDirectory(prefix='validator-fixture-',dir=HERE) as directory:
            path=Path(directory)
            (path/'cpt_train.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
            (path/'cpt_validation.jsonl').write_text('')
            counts={'rows':len(rows),'documents':1,'input_tokens':sum(len(r['input_ids']) for r in rows),
                'supervised_tokens':sum(sum(t!=-100 for t in r['labels']) for r in rows),
                'code_tokens':rows[0]['document_token_count']}
            manifest={'max_length':2048,'counts':{'cpt_train':counts},
                'artifacts':{n:{'sha256':c.sha(path/n)} for n in ('cpt_train.jsonl','cpt_validation.jsonl')}}
            (path/'manifest.json').write_text(json.dumps(manifest))
            result=subprocess.run([sys.executable,str(HERE/'validate_cpt_shard.py'),
                '--shard',str(path),'--documents',str(HERE/'profile-shard-v1/documents.jsonl'),
                '--output',str(path/'result.json')],capture_output=True,text=True,timeout=10,
                env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','CUDA_VISIBLE_DEVICES':'','TOKENIZERS_PARALLELISM':'false'})
            return result.returncode

    def test_retained_positive(self):self.assertEqual(self.run_fixture(),0)
    def test_nonterminal_eos_must_be_masked(self):
        self.assertNotEqual(self.run_fixture(lambda rows:rows[0]['labels'].__setitem__(-1,c.EOS)),0)
    def test_terminal_eos_must_be_supervised(self):
        self.assertNotEqual(self.run_fixture(lambda rows:rows[-1]['labels'].__setitem__(-1,-100)),0)
    def test_overlap_must_be_masked(self):
        self.assertNotEqual(self.run_fixture(lambda rows:rows[1]['labels'].__setitem__(1,rows[1]['input_ids'][1])),0)
    def test_changed_source_token_fails_hash_binding(self):
        def mutation(rows):rows[0]['input_ids'][1]=100;rows[0]['labels'][1]=100
        self.assertNotEqual(self.run_fixture(mutation),0)
    def test_wrong_partition_rejected(self):
        self.assertNotEqual(self.run_fixture(lambda rows:rows[0].__setitem__('cpt_partition','cpt_validation')),0)

if __name__=='__main__':unittest.main(verbosity=2)
