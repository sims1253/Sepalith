"""Card C13: the offline gate and R5 rules, and the RFT/DPO data selection, on synthetic inputs (stdlib only)."""
import json
import tempfile
import unittest
from pathlib import Path

from sepalith.training.offline import data, rules
from sepalith.training.sft import gate_rule
from sepalith.training.sft.campaign_edit_data import validate_row

TERMINAL = [97666, 24779, 651, 8089, 23877]
START = {"valid": 74, "cap": 1, "edit": 27, "nofp": 8, "score250": 117, "noedit": 1, "edit_total": 106}


def plain(value):
    """Untyped JSON copy of a result, for subscripting in assertions."""
    return json.loads(json.dumps(value))


def metrics(**changes):
    value = dict(START, loss=0.5)
    value.update(changes)
    return value


def outcome(rule, name):
    return next(row["outcome"] for row in rule["rows"] if row["name"] == name)


class OfflineGateTests(unittest.TestCase):
    def test_first_gate_uses_the_later_gate_table_against_the_starting_model(self):
        same = plain(rules.offline_gate(metrics(), START, first=True))
        self.assertEqual((same["outcome"], same["kind"], same["rule"]), ("continue", "offline_first_gate", "R1-offline"))
        self.assertIn("start_model", same["inputs"])
        # No "edit must beat the parent" floor: an unchanged edit count continues.
        self.assertEqual(rules.offline_gate(metrics(edit=27), START, first=True)["outcome"], "continue")
        self.assertEqual(rules.offline_gate(metrics(edit=24), START, first=True)["outcome"], "stop_for_user")
        self.assertEqual(rules.offline_gate(metrics(edit=19), START, first=True)["outcome"], "rollback")
        self.assertEqual(rules.offline_gate(metrics(cap=2), START, first=True)["outcome"], "stop_for_user")
        self.assertEqual(rules.offline_gate(metrics(score250=107), START, first=False)["outcome"], "rollback")
        self.assertEqual(rules.offline_gate(metrics(valid=67), START, first=False)["outcome"], "rollback")

    def test_noedit_dodge_stops_when_abstention_rises_without_better_edits(self):
        dodge = rules.offline_gate(metrics(noedit=4), START, first=True)
        self.assertEqual((dodge["outcome"], outcome(dodge, "noedit_dodge")), ("stop_for_user", "stop_for_user"))
        better = rules.offline_gate(metrics(noedit=4, edit_total=107), START, first=True)
        self.assertEqual(outcome(better, "noedit_dodge"), "continue")
        self.assertEqual(outcome(rules.offline_gate(metrics(noedit=3), START, first=True), "noedit_dodge"), "continue")

    def test_dpo_chosen_collapse_needs_a_steady_fall_and_no_score_gain(self):
        falling = [0.0] * 50 + [-0.1] * 50 + [-0.2] * 50 + [-0.3] * 50
        self.assertEqual(rules.block_means(falling + [5.0] * 10), [0.0, -0.1, -0.2, -0.3])
        stop = rules.offline_gate(metrics(), START, first=True, chosen_log_ratios=falling, start_score250=117)
        self.assertEqual(outcome(stop, "dpo_chosen_collapse"), "stop_for_user")
        gained = rules.offline_gate(metrics(score250=118), START, first=True, chosen_log_ratios=falling, start_score250=117)
        self.assertEqual(outcome(gained, "dpo_chosen_collapse"), "continue")
        bumpy = [0.0] * 50 + [-0.2] * 50 + [-0.1] * 50 + [-0.3] * 50
        self.assertEqual(outcome(rules.offline_gate(metrics(), START, first=True, chosen_log_ratios=bumpy, start_score250=117),
                                 "dpo_chosen_collapse"), "continue")
        positive = [0.4] * 50 + [0.3] * 50 + [0.2] * 50 + [0.1] * 50
        self.assertEqual(outcome(rules.offline_gate(metrics(), START, first=True, chosen_log_ratios=positive, start_score250=117),
                                 "dpo_chosen_collapse"), "continue")
        nan = rules.offline_gate(metrics(), START, first=True, chosen_log_ratios=[float("nan")], start_score250=117)
        self.assertEqual(nan["outcome"], "stop_for_user")

    def test_missing_counts_or_loss_stop_for_the_user(self):
        missing = dict(metrics())
        del missing["noedit"]
        self.assertEqual(rules.offline_gate(missing, START, first=True)["outcome"], "stop_for_user")
        self.assertEqual(rules.offline_gate(metrics(loss=float("nan")), START, first=True)["outcome"], "stop_for_user")

    def test_offline_metrics_reads_both_summaries(self):
        dev75 = {"counts": {"protocol_valid": 74, "cap_hit": 1, "edit_exact": 27, "strict_noop_false_suggestions": 8,
                            "predicted_noop": 24, "strict_noop_correct": 24}}
        dev250 = {"counts": {"edit_exact": 79, "strict_noop_correct": 38, "predicted_noop": 39}}
        self.assertEqual(rules.offline_metrics(dev75, dev250), START)


class R5Tests(unittest.TestCase):
    def test_three_outcomes(self):
        self.assertEqual(rules.apply_r5(metrics(score250=122), START)["outcome"], "adopt")
        self.assertEqual(rules.apply_r5(metrics(score250=122, nofp=9, valid=73), START)["outcome"], "adopt")
        self.assertEqual(rules.apply_r5(metrics(score250=118), START)["outcome"], "stop_for_user")
        self.assertEqual(rules.apply_r5(metrics(score250=121), START)["outcome"], "stop_for_user")
        self.assertEqual(rules.apply_r5(metrics(score250=117), START)["outcome"], "discard")
        self.assertEqual(rules.apply_r5(metrics(score250=130, nofp=10), START)["outcome"], "discard")
        self.assertEqual(rules.apply_r5(metrics(score250=130, valid=72), START)["outcome"], "discard")
        self.assertEqual(rules.apply_r5({"score250": 120}, START)["outcome"], gate_rule.STOP_FOR_USER)


def train_row(rid, family, body, prompt=(5, 6, 7)):
    ids = [0, *prompt, *body, *TERMINAL, 1]
    return {"id": rid, "family": family, "package_id": "p", "split": "train", "renderer_id": "zeta2-prm03-v1",
            "tokenizer_json_sha256": "3e065a558a034185fe299917b398685c1facd0169a9eea1e629eb30c171fed81",
            "input_ids": ids, "target_start": len(prompt) + 1, "prompt_token_count": len(prompt),
            "target_body_tokens": list(body), "target_terminal_tokens": TERMINAL, "target_body_token_count": len(body),
            "target_terminal_token_count": len(TERMINAL), "target_token_count": len(body) + len(TERMINAL),
            "target_operation": "no_op" if family == "no_op" else "replace", "bos_token_id": 0, "eos_token_id": 1,
            "target_text": "x\n>>>>>>> UPDATED", "target_body_text": "x",
            "prompt_text": "head\n<<<<<<< CURRENT\na <- 1\nb <- 2\n=======\n<[fim-middle]>\n"}


class SelectionTests(unittest.TestCase):
    def test_current_region_and_smallest_diff_ranking(self):
        prompt = "x\n<<<<<<< CURRENT\na <- 1\nb <- <|user_cursor|>2\n=======\n<[fim-middle]>\n"
        self.assertEqual(data.current_region(prompt), "a <- 1\nb <- 2")
        self.assertIsNone(data.current_region("no region"))
        self.assertEqual(data.completion_body_text("a\nb\n>>>>>>> UPDATED"), "a\nb")
        samples = [
            {"sample": 0, "generated_ids": [1, 2, 3], "text": "a <- 1\nb <- 3\nc <- 4\n>>>>>>> UPDATED", "exact_raw": False},
            {"sample": 1, "generated_ids": [1, 2], "text": "a <- 1\nb <- 3\n>>>>>>> UPDATED", "exact_raw": True},
            {"sample": 2, "generated_ids": [1, 2], "text": "a <- 1\nb <- 3\n>>>>>>> UPDATED", "exact_raw": True},
            {"sample": 3, "generated_ids": [9, 9], "text": "a <- 1\nb <- 3\n>>>>>>> UPDATED", "exact_raw": False},
        ]
        ranked = plain(data.rank_correct_samples(samples, "a <- 1\nb <- 2"))
        self.assertEqual([s["sample"] for s in ranked], [1, 3, 0])  # distinct ids; smallest diff, then raw gold identity
        self.assertEqual([s["diff_lines"] for s in ranked], [1, 1, 2])

    def test_family_caps_use_train_share_of_the_pool_not_the_scarcest_family(self):
        caps = data.family_caps({"a": 50, "b": 30, "c": 20}, {"a": 10, "b": 100, "c": 100}, 100)
        self.assertEqual(caps, {"a": 10, "b": 30, "c": 20})
        # The census numbers: 7,528-sample pool over the 15,006 TRAIN rows.
        train = {"finish_block": 7785, "format_propagation": 1615, "na_rm_propagation": 136, "no_op": 1094,
                 "pipe_rewrite": 1983, "rename_propagation": 1578, "roxygen_drafting": 815}
        supply = {"finish_block": 1181, "format_propagation": 1505, "na_rm_propagation": 137, "no_op": 1100,
                  "pipe_rewrite": 1995, "rename_propagation": 1589, "roxygen_drafting": 21}
        caps = data.family_caps(train, supply, 7528)
        self.assertEqual(caps, {"finish_block": 1181, "format_propagation": 810, "na_rm_propagation": 68, "no_op": 548,
                                "pipe_rewrite": 994, "rename_propagation": 791, "roxygen_drafting": 21})
        self.assertEqual(sum(caps.values()), 4413)

    def test_select_prefers_low_pass_rate_prompts_and_best_samples_first(self):
        pool = {"p1": [{"sample": 0}, {"sample": 1}], "p2": [{"sample": 4}], "p3": [{"sample": 2}, {"sample": 5}]}
        family = {"p1": "f", "p2": "f", "p3": "f"}
        rate = {"p1": 1.0, "p2": 0.25, "p3": 0.5}
        chosen = plain(data.select_rft(pool, family, rate, {"f": 4}))
        self.assertEqual([(c["id"], c["sample"], c["round"]) for c in chosen],
                         [("p2", 4, 1), ("p3", 2, 1), ("p1", 0, 1), ("p3", 5, 2)])
        with self.assertRaisesRegex(ValueError, "supply below its cap"):
            data.select_rft(pool, family, rate, {"f": 6})

    def test_rft_row_keeps_the_sft_render_and_termination(self):
        row = train_row("r1", "finish_block", [11, 12])
        sample = {"sample": 3, "generated_ids": [11, 13, *TERMINAL, 1], "text": "y\n>>>>>>> UPDATED", "round": 1, "pass_rate": 0.5,
                  "diff_lines": 1, "diff_basis": "current_region", "exact_raw": False, "comparison": "air"}
        out = plain(data.rft_row(row, sample, "m"))
        self.assertEqual(out["id"], "r1:rft:s3")
        self.assertEqual(out["input_ids"], [0, 5, 6, 7, 11, 13, *TERMINAL, 1])
        self.assertEqual(out["target_body_text"], "y")
        self.assertFalse(out["rft"]["same_tokens_as_gold"])
        validate_row(out, 4096)  # the SFT row contract: split, renderer, BOS/EOS, body + terminal + EOS
        with self.assertRaisesRegex(ValueError, "end with EOS 1"):
            data.rft_row(row, dict(sample, generated_ids=[11, *TERMINAL, 130073]), "m")
        with self.assertRaisesRegex(ValueError, "protocol terminal"):
            data.rft_row(row, dict(sample, generated_ids=[11, 12, 1]), "m")

    def test_train_mix_excludes_rft_prompts_and_is_seeded(self):
        ids = [f"r{i}" for i in range(100)]
        first = data.train_mix_ids(ids, {"r1", "r2"}, 30)
        self.assertEqual(first, data.train_mix_ids(reversed(ids), {"r1", "r2"}, 30))
        self.assertEqual(len(first), 30)
        self.assertFalse({"r1", "r2"} & set(first))
        self.assertEqual(data.train_mix_count(4413), 1891)  # 30% of the cohort

    def test_pair_rows_normalize_termination_and_check_gold(self):
        row = train_row("r1", "finish_block", [11, 12])
        candidate = {"id": "r1", "family": "finish_block", "expected_operation": "replace",
                     "chosen": {"source": "gold", "sample": None, "generated_ids": [11, 12, *TERMINAL, 1], "text": None},
                     "rejected": {"sample": 2, "generated_ids": [14, *TERMINAL, 130073], "text": "z", "outcome": "near_miss",
                                  "reward": 0.1, "parse": "passed"}}
        pair, fixed = data.pair_row(candidate, 0, row, "m")
        pair = plain(pair)
        self.assertEqual(fixed, 1)
        self.assertEqual((pair["pair_id"], pair["prompt_ids"]), ("r1:p0", [0, 5, 6, 7]))
        self.assertEqual(pair["rejected_ids"], [14, *TERMINAL, 1])
        self.assertEqual(pair["chosen"]["text"], row["target_text"])
        bad = json.loads(json.dumps(candidate))
        bad["chosen"]["generated_ids"] = [11, *TERMINAL, 1]
        with self.assertRaisesRegex(ValueError, "gold chosen differs"):
            data.pair_row(bad, 0, row, "m")
        same = json.loads(json.dumps(candidate))
        same["rejected"]["generated_ids"] = [11, 12, *TERMINAL, 1]
        with self.assertRaisesRegex(ValueError, "chosen equals rejected"):
            data.pair_row(same, 0, row, "m")

    def test_pair_schedule_is_one_pass_with_a_completed_last_update(self):
        ids = [f"p{i}" for i in range(37)]
        plan = plain(data.pair_schedule(ids, "sha"))
        self.assertEqual((plan["max_steps"], plan["draw_count"], plan["replays"], plan["gate_updates"]), (3, 48, 11, [2, 3]))
        draws = plan["draws"]
        self.assertEqual(sorted(draws[:37]), list(range(37)))
        self.assertEqual(draws[37:], draws[:11])
        self.assertFalse(set(draws[32:37]) & set(draws[37:]))
        self.assertEqual(data.halves(1142), [571, 1142])
        self.assertEqual(data.halves(435), [218, 435])


class BuildTests(unittest.TestCase):
    def test_build_end_to_end_on_a_small_census(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = [train_row(f"e{i:02d}", "finish_block", [20 + i % 7, 30 + i]) for i in range(40)]
            rows += [train_row(f"n{i:02d}", "no_op", [80, 12991], prompt=(5, 6, 7, 100 + i)) for i in range(12)]
            for r in rows:
                if r["family"] != "no_op":
                    r["input_ids"][2] = 200 + int(r["id"][1:])  # distinct prompts
            train = root / "train.jsonl"
            train.write_text("".join(json.dumps(r) + "\n" for r in rows))
            census = root / "census"
            census.mkdir()
            prompts, rates, scores, pairs = [], [], [], []
            for r in rows:
                prompt = r["input_ids"][:r["target_start"]]
                prompts.append({"id": r["id"], "prompt_sha256": data.sha256_ids(prompt)})
                correct = int(r["id"][1:]) % 3 != 0
                rates.append({"id": r["id"], "exact_count": 8 if correct else 0, "k": 8})
                gold = r["input_ids"][r["target_start"]:]
                scores.append({"id": r["id"], "sample": 0, "exact": correct, "protocol_valid": True, "cap_hit": False,
                               "generated_ids": gold, "text": "a <- 1\n>>>>>>> UPDATED", "exact_raw": True, "comparison": "air"})
                if not correct:
                    pairs.append({"id": r["id"], "family": r["family"], "expected_operation": r["target_operation"],
                                  "chosen": {"source": "gold", "sample": None, "generated_ids": gold, "text": None},
                                  "rejected": {"sample": 1, "generated_ids": [99, *TERMINAL, 1], "text": "q", "outcome": "near_miss",
                                               "reward": 0.0, "parse": "passed"}})
            files = {}
            for name, values in (("census-prompts.jsonl", prompts), ("pass-rates.jsonl", rates), ("scores.jsonl", scores),
                                 ("dpo-pair-candidates.jsonl", pairs)):
                (census / name).write_text("".join(json.dumps(v) + "\n" for v in values))
                files[name] = {"sha256": data.sha256(census / name)}
            (census / "census-summary.json").write_text(json.dumps({"files": files, "final_set_access": False}))
            manifest = plain(data.build(census_dir=census, census_summary_sha256=data.sha256(census / "census-summary.json"), model_id="m",
                                  output=root / "out", train_rows=train, train_rows_sha256=data.sha256(train)))
            self.assertEqual(manifest["rft"]["caps_by_family"], {"finish_block": 26, "no_op": 7})
            self.assertEqual(manifest["cohort"]["rft_rows"], 33)
            self.assertEqual(manifest["cohort"]["train_mix_rows"], 14)
            self.assertEqual(manifest["schedule"]["noop_share"], 0.1875)
            self.assertTrue(all(v for v in manifest["schedule"]["checks"].values() if isinstance(v, bool)))
            self.assertEqual(manifest["dpo"]["pairs"], len(pairs))
            for record in manifest["outputs"].values():
                self.assertEqual(data.sha256(record["path"]), record["sha256"])
            with self.assertRaisesRegex(ValueError, "fresh"):
                data.build(census_dir=census, census_summary_sha256=data.sha256(census / "census-summary.json"), model_id="m",
                           output=root / "out", train_rows=train, train_rows_sha256=data.sha256(train))


if __name__ == "__main__":
    unittest.main()
