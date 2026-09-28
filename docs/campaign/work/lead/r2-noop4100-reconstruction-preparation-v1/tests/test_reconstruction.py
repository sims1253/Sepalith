import copy, hashlib, json, sys, tempfile, unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'source'))
import prepare_noop4100 as subject

BASE=Path('/mnt/e/sepalith/campaign-20260915/data-work')
DECISIONS=BASE/'Sourcewalk-noop-expansion-v4/recovery-decisions.jsonl'
REPLAY=BASE/'Sourcewalk-independent-replay-v3/full-01/shards'
PACKETS=BASE/'DAT10-novel-v1/source-walk-shards-v1'

def find(path,row_id):
    with path.open() as stream:
        for line in stream:
            row=json.loads(line)
            if (row.get('row_id') or row.get('row_ref',{}).get('row_id'))==row_id:return row
    raise KeyError(row_id)

def joined(row_id):
    decision=find(DECISIONS,row_id); shard=decision['shard']
    ledger=find(REPLAY/f'shard-{shard:04d}/ledger.jsonl',row_id)
    packet=find(PACKETS/f'shard-{shard:04d}/structured-materialization-v1/candidate-packets.jsonl',row_id)
    return decision,ledger,packet

class Reconstruction(unittest.TestCase):
    def test_real_nonempty_unchanged_selection_reconstructs_target_free_full_source(self):
        decision,ledger,packet=joined('47a96cc61e9860688ea9114e')
        prediction,sidecar=subject.recovered_output(packet,ledger,decision)
        raw=Path(packet['validation']['source_path']).read_bytes()
        self.assertEqual(prediction['preedit_text'].encode(),raw)
        self.assertEqual(hashlib.sha256(raw).hexdigest(),prediction['preedit_sha256'])
        self.assertTrue(sidecar['recorded_unchanged_selection_nonempty'])
        self.assertFalse(any('target' in key.lower() or 'gold' in key.lower() for key in prediction))
        self.assertEqual(prediction['cursor']['line'],decision['geometry']['global_cursor']['line'])

    def test_real_uniform_crlf_preserves_raw_bytes_and_projects_cursor(self):
        decision,ledger,packet=joined('49da0198cb6f200c0214a9ee')
        prediction,sidecar=subject.recovered_output(packet,ledger,decision)
        raw=Path(packet['validation']['source_path']).read_bytes()
        self.assertIn(b'\r\n',raw);self.assertEqual(prediction['document_eol'],'crlf')
        self.assertEqual(prediction['preedit_text'].encode(),raw)
        self.assertEqual(sidecar['identity']['cursor_projection']['window_start_utf16_column'],0)

    def test_nonempty_target_must_equal_recorded_region(self):
        decision,ledger,packet=joined('47a96cc61e9860688ea9114e')
        broken=copy.deepcopy(packet);broken['result']['target_body']=[]
        with self.assertRaisesRegex(ValueError,'recovery no longer valid'):
            subject.recovered_output(broken,ledger,decision)

    def test_missing_source_is_infrastructure_failure_not_row_hold(self):
        decision,ledger,packet=joined('47a96cc61e9860688ea9114e')
        broken=copy.deepcopy(packet);broken['validation']['source_path']='/definitely/missing/noop-source.R'
        with self.assertRaises(FileNotFoundError):subject.recovered_output(broken,ledger,decision)

    def test_utf16_projection_includes_window_prefix_and_rejects_surrogate_split(self):
        raw='prefix😀WINDOW\nend'.encode(); packet={'result':{'selection_source':{'text':'WINDOW\nend'},'context':{'replacement_range':{'start':{'line':0,'character':1},'end':{'line':0,'character':2}},'region_old':['I']}}}
        provenance={'source_window_occurrence_method':'exact_bytes'}
        cursor,selected,proof=subject.project_cursor_and_selection(packet,provenance,raw)
        self.assertEqual(selected,'I');self.assertEqual(cursor,{'line':0,'character':9});self.assertFalse(proof['window_starts_at_line_boundary'])
        with self.assertRaisesRegex(ValueError,'surrogate'):subject.utf16_column_offset('😀',1)

    def test_mixed_eol_decision_remains_named_hold(self):
        with DECISIONS.open() as stream:
            decision=next(json.loads(line) for line in stream if json.loads(line).get('status')=='hold')
        rid=decision['row_id'];shard=decision['shard'];ledger=find(REPLAY/f'shard-{shard:04d}/ledger.jsonl',rid);packet=find(PACKETS/f'shard-{shard:04d}/structured-materialization-v1/candidate-packets.jsonl',rid)
        item={'row_id':rid,'shard':shard,'status':subject.SUPPORTED}
        p,s,hold=subject.candidate_output(item,packet,ledger,{rid:decision})
        self.assertIsNone(p);self.assertIsNone(s);self.assertEqual(hold['status'],'hold');self.assertTrue(hold['recovery_decision_bound'])

    def test_duplicate_geometries_are_retained(self):
        row={'row_id':'a','preedit_sha256':'f'*64,'cursor':{'line':1,'character':0},'path':'R/x.R','workspace_root':'/a','absolute_document_path':'/a/R/x.R'}
        other={**row,'row_id':'b','workspace_root':'/b','absolute_document_path':'/b/R/x.R'}
        groups=subject.base.validate_prediction_set([row,other])
        self.assertEqual(groups[0]['row_ids'],['a','b']);self.assertEqual(groups[0]['disposition'],'retain_all_until_provider_prompt_target_dedup')

if __name__=='__main__':unittest.main(verbosity=2)
