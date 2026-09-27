import copy, unittest
from semantic_gate import classify_evidence, summarize


def valid():
    terminal = [30, 40]
    return {
        "row_id": "train-row",
        "source_sha256_matches": True,
        "raw_line_sha256_matches": True,
        "global_train_split_verified": True,
        "protected_split_disjoint": True,
        "direct_license_verified": True,
        "before_after_inverse_roundtrip": True,
        "unique_target_definition": True,
        "target_range_matches": True,
        "source_parses": True,
        "complete_target": True,
        "prompt_excludes_target": True,
        "required_spans_retained": True,
        "documented_params_duplicate": False,
        "documented_params_missing_from_formals": [],
        "unsupported_documentation_claims": [],
        "reference_occurrences": [{"name":"i", "category":"local_binding_reference"}],
        "token_row": {"input_ids":[0,10,20,30,40,1],"target_start":2,"target_body_tokens":[20],"target_terminal_tokens":terminal,"canonical_terminal_tokens":terminal,"prompt_token_count":1,"target_token_count":3,"bos_token_id":0,"eos_token_id":1,"split":"train"},
    }


class SemanticGateTest(unittest.TestCase):
    def test_local_loop_binding_is_supported(self):
        self.assertEqual(classify_evidence(valid())["status"], "supported_candidate_root_review_required")

    def test_occurrence_backed_nse_column_is_supported(self):
        e=valid(); e["reference_occurrences"]=[{"name":"group","category":"nse_data_column","call_head":"subset","call_span_sha256":"a"*64,"data_expression_sha256":"b"*64}]
        self.assertEqual(classify_evidence(e)["status"], "supported_candidate_root_review_required")

    def test_name_only_nse_allowlist_is_rejected(self):
        e=valid(); e["reference_occurrences"]=[{"name":"group","category":"nse_data_column"}]
        self.assertIn("nse_occurrence_evidence_incomplete", classify_evidence(e)["review_reasons"])

    def test_true_unbound_global_remains_held(self):
        e=valid(); e["reference_occurrences"]=[{"name":"LABEL","category":"unbound_or_unsupported"}]
        self.assertEqual(classify_evidence(e)["status"], "hold_semantic_evidence_required")

    def test_invented_param_is_hard_hold(self):
        e=valid(); e["documented_params_missing_from_formals"]=["invented"]
        self.assertIn("invented_or_wrong_param", classify_evidence(e)["blockers"])

    def test_hidden_target_leakage_is_hard_hold(self):
        e=valid(); e["prompt_excludes_target"]=False
        self.assertIn("hidden_target_leakage", classify_evidence(e)["blockers"])

    def test_missing_helper_span_is_hard_hold(self):
        e=valid(); e["required_spans_retained"]=False
        self.assertIn("necessary_context_span_missing", classify_evidence(e)["blockers"])

    def test_special_tokens_and_vocab_are_canonical(self):
        e=valid(); e["token_row"]["input_ids"][2]=130560
        self.assertIn("invalid_token_id", classify_evidence(e)["blockers"])

    def test_unsupported_prose_claim_is_review_queue(self):
        e=valid(); e["unsupported_documentation_claims"]=[{"claim_hash":"c"*64}]
        self.assertEqual(classify_evidence(e)["status"], "hold_semantic_evidence_required")

    def test_denominator_is_preserved(self):
        e=valid(); result=classify_evidence(e)
        self.assertEqual(summarize([result,result])["rows"],2)


if __name__ == "__main__": unittest.main()
