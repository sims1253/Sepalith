#!/usr/bin/env python3
import json
from pathlib import Path
import sys
import unittest

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
from campaign_cpt_data import CptDataError, validate_draw_schedule, validate_materialized_rows

ROWS = Path('/mnt/e/sepalith/campaign-20260915/data-work/CPT-remaining-v1/cpt_train.jsonl')
RECIPE = Path('/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb/docs/campaign/work/lead/r2-cpt-remaining-v1/recipe.json')


def later_chunk():
    with ROWS.open() as stream:
        for line in stream:
            row = json.loads(line)
            if row['chunk_index'] > 0:
                return row
    raise AssertionError('fixture lacks a later source chunk')


class RemainingContractTest(unittest.TestCase):
    def test_sparse_remainder_accepts_a_named_later_chunk(self):
        summary = validate_materialized_rows([later_chunk()], require_complete_documents=False)
        self.assertEqual(summary['rows'], 1)
        self.assertTrue(summary['documents_detail'][0]['partial'])

    def test_ordinary_complete_document_contract_rejects_later_chunk_alone(self):
        with self.assertRaisesRegex(CptDataError, 'start at zero'):
            validate_materialized_rows([later_chunk()])

    def test_sparse_remainder_rejects_duplicate_chunk(self):
        row = later_chunk()
        duplicate = dict(row)
        duplicate['row_id'] = 'different-row-id-for-same-chunk'
        with self.assertRaisesRegex(CptDataError, 'duplicate chunk index'):
            validate_materialized_rows([row, duplicate], require_complete_documents=False)

    def test_source_order_schedule_and_named_replay_are_exact(self):
        rows = [
            {'id': f'r{i}', 'input_ids': [0, i + 2, 1], 'labels': [-100, i + 2, 1],
             'attention_mask': [1, 1, 1]} for i in range(13)
        ]
        schedule = {'split_id': 'train', 'method': 'one_pass_source_order_plus_named_replay_v1',
                    'seed': 3407, 'max_steps': 1, 'effective_batch': 16,
                    'token_rows_sha256': 'a' * 64,
                    'row_ids': [f'r{i}' for i in range(13)] + ['r0', 'r1', 'r2'],
                    'replay_row_ids': ['r0', 'r1', 'r2']}
        self.assertEqual(validate_draw_schedule(schedule, rows, token_rows_sha256='a' * 64,
                                                max_steps=1)['draws'], 16)
        schedule['row_ids'][0], schedule['row_ids'][1] = schedule['row_ids'][1], schedule['row_ids'][0]
        with self.assertRaisesRegex(CptDataError, 'Source-order pass'):
            validate_draw_schedule(schedule, rows, token_rows_sha256='a' * 64, max_steps=1)

    def test_recipe_has_fresh_optimizer_and_every_full_cadence(self):
        recipe = json.loads(RECIPE.read_text())
        self.assertIsNone(recipe['resume_from'])
        self.assertNotIn('resume_binding', recipe)
        every = recipe['checkpoint']['full_every']
        self.assertEqual(recipe['parameters']['max_steps'] % every, 0)
        self.assertEqual(list(range(every, recipe['parameters']['max_steps'] + 1, every)),
                         [317, 634, 951, 1268, 1585, 1902])


if __name__ == '__main__':
    unittest.main()
