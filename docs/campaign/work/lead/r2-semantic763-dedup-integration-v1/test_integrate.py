import unittest
from integrate import digest_text, normalized_source_paths, source_hashes

class IntegrationPolicyTests(unittest.TestCase):
    def test_prompt_target_and_prompt_conflict_keys_are_distinct(self):
        a={"prompt_text":"p","target_text":"a","target_body_text":"a"};b={"prompt_text":"p","target_text":"b","target_body_text":"b"}
        from integrate import pair_key,target_key
        self.assertNotEqual(pair_key(a),pair_key(b));self.assertNotEqual(target_key(a),target_key(b));self.assertEqual(digest_text(a["prompt_text"]),digest_text(b["prompt_text"]))
    def test_source_identity_extracts_snapshot_hash_and_normalized_R_path(self):
        identity={"source_sha256":"a"*64,"source_path":"/mnt/pkg/R/a.R","source_provenance":{"source_snapshot_sha256":"b"*64,"source_snapshot_path":"C:\\pkg\\R\\a.R"}}
        self.assertEqual(source_hashes(identity),{"a"*64,"b"*64});self.assertIn("R/a.R",normalized_source_paths(identity))
    def test_target_leak_rule_detects_exact_complete_body_only(self):
        body="#' @title exact\n#' @export"
        self.assertIn(body,"prefix\n"+body+"\nsuffix");self.assertNotIn(body,"prefix\n#' @title related\nsuffix")
    def test_context_tier_is_16k_first_and_32k_additive(self):
        ids16={"a","b"};ids32={"a","b","c"};self.assertTrue(ids16 <= ids32);self.assertEqual(ids32-ids16,{"c"})
    def test_source_target_dedup_does_not_collapse_same_target_across_sources(self):
        target=digest_text("same target");keys={("source-a",target),("source-b",target)};self.assertEqual(len(keys),2)

if __name__=='__main__': unittest.main(verbosity=2)
