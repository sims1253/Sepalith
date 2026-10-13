import unittest

from campaign_token_audit import length_profiles, rendered_collisions, selected_context, text_digest
from sepalith.campaign_protocol import SCHEMA_VERSION, render_prompt


def source_result():
    text = 'before <- "😀"\r\ncurrent <- 1\r\nafter <- 2'
    return {
        'context': {
            'schema_version': SCHEMA_VERSION, 'path': 'R/example.R',
            'prefix': ['before <- "😀"'], 'selected_references': [], 'history': [],
            'diagnostics': [], 'retrieval': [], 'scope_mode': 'off', 'scope_lines': [],
            'suffix_lines': ['after <- 2'], 'region_old': ['current <- 1'],
            'cursor': {'region_line_index': 0, 'code_point_column': 12, 'utf16_column': 12},
            'replacement_range': {
                'uri': 'file:///example.R', 'document_version': 2,
                'content_sha256': text_digest(text),
                'start': {'line': 1, 'character': 0}, 'end': {'line': 1, 'character': 12},
            }, 'document_eol': 'crlf',
        },
        'selection_source': {'text': text, 'content_sha256': text_digest(text),
                             'region_start_line': 1, 'region_end_line': 1,
                             'availability': 'full_snapshot'},
    }


class TokenAuditTest(unittest.TestCase):
    def test_source_bytes_and_required_region_survive_overflow(self):
        context, selection = selected_context(source_result(), max_source_utf16=3)
        self.assertEqual(context.region_old, ('current <- 1',))
        self.assertTrue(selection['required_overflow'])
        self.assertTrue(selection['support_revalidation_required'])
        self.assertEqual(context.document_eol, 'crlf')
        self.assertIn('current <- 1', render_prompt(context).replace('<|user_cursor|>', ''))

    def test_stale_or_partial_line_selection_is_excluded(self):
        result = source_result()
        result['selection_source']['text'] += '\r\nchanged'
        with self.assertRaisesRegex(ValueError, 'hash_mismatch'):
            selected_context(result)
        result = source_result()
        result['context']['replacement_range']['start']['character'] = 1
        with self.assertRaises(ValueError):
            selected_context(result)

    def test_completion_document_text_is_hash_checked_and_alias_conflicts_fail(self):
        result = source_result()
        source = result['selection_source']
        source['document_text'] = source.pop('text')
        context, _ = selected_context(result)
        self.assertEqual(context.region_old, ('current <- 1',))
        source['text'] = source['document_text'] + 'changed'
        with self.assertRaisesRegex(ValueError, 'aliases_disagree'):
            selected_context(result)
        source.pop('text')
        source['document_text'] += 'changed'
        with self.assertRaisesRegex(ValueError, 'hash_mismatch'):
            selected_context(result)

    def test_canonical_empty_region_preserves_physical_blank_line_and_eof(self):
        for text in ('before <- 1\n\nafter <- 2', 'before <- 1\n'):
            result = source_result()
            result['context'].update(
                prefix=['before <- 1'], region_old=[], document_eol='lf',
                cursor={'region_line_index': -1, 'code_point_column': None, 'utf16_column': None},
                suffix_lines=['after <- 2'] if text.endswith('after <- 2') else [],
            )
            result['context']['replacement_range'].update(
                content_sha256=text_digest(text),
                start={'line': 1, 'character': 0}, end={'line': 1, 'character': 0},
            )
            result['selection_source'].update(text=text, content_sha256=text_digest(text))
            context, selection = selected_context(result)
            self.assertEqual(context.region_old, ())
            self.assertEqual(context.replacement_range.start.line, 1)
            self.assertEqual(selection['region'], [''])
            self.assertEqual(context.prefix, ('before <- 1',))
        result['selection_source']['text'] = 'before <- 1'
        with self.assertRaises(ValueError):
            selected_context(result)

    def test_profiles_count_terminal_eos_and_never_shorten_target(self):
        row = {'input_ids': list(range(2241)), 'target_start': 2048,
               'prompt_token_count': 2047, 'target_token_count': 192,
               'target_body_token_count': 186}
        result = length_profiles(row)
        self.assertEqual(result['response_with_terminal_eos'], 193)
        self.assertFalse(result['rl_prompt2048_response192'])
        self.assertFalse(result['sft_2048'])
        self.assertTrue(result['sft_4096'])
        self.assertEqual(len(row['input_ids']), 2241)

    def test_rendered_collisions_ignore_hidden_identity_metadata(self):
        context, _ = selected_context(source_result())
        prompt = text_digest(render_prompt(context))
        candidates = [
            {'prompt_sha256': prompt, 'target_sha256': text_digest(target),
             'row': {'id': identifier, 'split': split}, 'source_ref': {'group_id': group}}
            for identifier, split, group, target in [
                ('a', 'train', 'g1', 'current <- 2'),
                ('b', 'dev', 'g2', 'current <- 3'),
                ('c', 'train', 'g1', 'current <- 2'),
            ]
        ]
        rows = rendered_collisions(candidates)
        self.assertEqual(len(rows), 1)
        self.assertEqual(len(rows[0]['members']), 3)
        self.assertTrue(rows[0]['different_targets'])
        self.assertTrue(rows[0]['cross_split'])


if __name__ == '__main__':
    unittest.main()
