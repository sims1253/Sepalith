"""Execution-grounded reward for exec-env tasks (policy ``sepalith.reward.exec.v1``).

A drop-in alternative to ``training.grpo.trainer.PoolReward``: ``ExecReward`` is
callable as ``reward(ids, completions) -> list[record]`` with the same record
keys the GRPO trainer reads (``reward``, ``exact``, ``outcome``). It is opt-in:
nothing in the trainer, its default reward (v2.1) or any bound recipe changes.
Legacy TRAIN-row replay keeps reward v2.1; this reward is family-specific and
only scores exec-env tasks.

Pipeline per completion:

1. protocol (as reward v2.1): token checks (canonical EOS, no control token
   before it, cap) and ``parse_output`` against the task's PRM03 context;
2. splice: the parsed body replaces the region bytes in the task's executable
   file (``exec.before`` + body + ``exec.after``); nothing outside the region
   can change;
3. static screen: constructs that can tamper with the harness or the tests
   (stack and environment inspection, ``quit``, ``assign``/``<<-``, dynamic
   eval, I/O, ``Sys.*``, namespace access, ``setTimeLimit``, ...) are refused
   unless the original file uses them too; ``rename_propagation`` also requires
   the old name to be gone from the region;
4. harness (sandboxed fork per job): parse, structure check (the top-level
   assignments are exactly the expected helpers and function, so closing the
   function early and adding code is caught), source, then the complete
   hidden suite; assertions are counted by the helpers, never from exit status.

| Outcome | Reward |
| --- | --- |
| Protocol-invalid output or cap hit | -0.5 |
| Edit case answered with no edit | 0 |
| Spliced file does not parse / breaks structure / forbidden construct / does not source | 0 |
| Timeout, crash, ``quit()`` of the candidate | 0 |
| Otherwise | ``w_behaviour * B + w_all * A (+ exact_bonus * exact_match)``, capped at 1 |

``B`` is the behaviour-weighted pass fraction: per category, passed assertions
over the task's expected count (assertions that never ran count as failed),
then averaged over categories. ``A`` is 1 when every expected assertion ran and
passed. Defaults: ``w_behaviour = w_all = 0.5``, ``exact_bonus = 0``; exact
textual match is logged separately (``exact_match``). The record's ``exact`` is
``A`` (functional correctness), which is what the trainer counts as a pass.

Infrastructure faults (a worker process dying, garbled protocol) are retried by
the harness and then raised; they are never scored as candidate failures.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import json
from pathlib import Path
import re

from sepalith.campaign_protocol import VOCAB_SIZE, PromptContext, ProtocolError, parse_output
from sepalith.exec_env.harness import ExecHarness, ExecJob, ExecResult
from sepalith.training.rl.air_canonical import normalize_lines

POLICY_ID = "sepalith.reward.exec.v1"
RECORD_SCHEMA = "sepalith.reward-exec.record.v1"
REWARD_PROTOCOL_INVALID = -0.5
REWARD_CAP_HIT = -0.5

FORBIDDEN_IN_COMPLETION = re.compile(
    r"(?<![\w.])(quit|q|sys\.frames?|sys\.function|sys\.calls|parent\.frame|parent\.env|environment|"
    r"globalenv|emptyenv|topenv|as\.environment|assign|get0?|mget|exists|eval|evalq|eval\.parent|parse|"
    r"body|formals|trace|untrace|lockBinding|unlockBinding|lockEnvironment|makeActiveBinding|"
    r"setTimeLimit|setSessionTimeLimit|Sys\.[A-Za-z]+|system2?|file|url|readLines|writeLines|"
    r"readRDS|saveRDS|source|sys\.source|sink|library|require|requireNamespace|loadNamespace|"
    r"getNamespace|asNamespace|attach|options|invokeRestart|\.Internal|\.Primitive|\.Call|\.External|"
    r"reg\.finalizer|signalCondition|mcparallel|mcfork|sendMaster)\s*\(|<<-|->>|:::|@\s*\.xData")


@dataclass(frozen=True)
class ExecRewardConfig:
    w_behaviour: float = 0.5
    w_all: float = 0.5
    exact_bonus: float = 0.0
    max_completion_tokens: int = 1024
    # completions much longer than the reference remainder are scaled down: code that is dead under the
    # suite (extra branches, attribute writes from fields fake inputs lack) would otherwise pass for free
    overlong_factor: float = 2.0
    overlong_slack: int = 6
    overlong_scale: float = 0.5

    def __post_init__(self) -> None:
        if min(self.w_behaviour, self.w_all, self.exact_bonus) < 0 or self.w_behaviour + self.w_all <= 0:
            raise ValueError("reward weights must be non-negative and not all zero")


@dataclass(frozen=True)
class ExecTask:
    id: str
    family: str
    context: PromptContext
    target_body_text: str
    before: str
    after: str
    expect: tuple[str, ...]
    timeout: float
    suite: str
    expected_categories: dict[str, int]
    gold_text: str
    forbid_in_region: str | None = None

    @classmethod
    def from_dict(cls, row: Mapping[str, object]) -> ExecTask:
        exec_ = row["exec"]
        assert isinstance(exec_, Mapping)
        return cls(str(row["id"]), str(row["family"]), PromptContext.from_mapping(row["context"]),
                   str(row["target_body_text"]), str(exec_["before"]), str(exec_["after"]),
                   tuple(exec_["expect"]), float(exec_["timeout"]), str(row["suite"]),
                   {str(k): int(v) for k, v in dict(row["expected_categories"]).items()},  # type: ignore[call-overload]
                   str(exec_.get("gold_text", "")), exec_.get("forbid_in_region"))  # type: ignore[arg-type]

    def splice(self, body: Sequence[str]) -> str:
        parts = [p for p in (self.before, "\n".join(body), self.after) if p != ""]
        return "\n".join(parts) + "\n"


def load_tasks(path: Path) -> dict[str, ExecTask]:
    out = {}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                task = ExecTask.from_dict(json.loads(line))
                out[task.id] = task
    return out


def _code_only(text: str) -> str:
    from sepalith.data.synth.common import STRING_RE, strip_comment
    return "\n".join(STRING_RE.sub('""', strip_comment(line)) for line in text.split("\n"))


def forbidden_constructs(body: Sequence[str], gold_text: str) -> list[str]:
    """Tamper-capable constructs in the completion that the original file does not use."""
    allowed = {m.group(0).replace(" ", "") for m in FORBIDDEN_IN_COMPLETION.finditer(_code_only(gold_text))}
    found = {m.group(0).replace(" ", "") for m in FORBIDDEN_IN_COMPLETION.finditer(_code_only("\n".join(body)))}
    return sorted(found - allowed)


def score_result(task: ExecTask, result: ExecResult, config: ExecRewardConfig) -> tuple[float, str, dict]:
    info: dict = {"parse": result.parse_ok, "structure": result.structure_ok, "sourced": result.sourced_ok,
                  "error_class": result.error_class, "elapsed": round(result.elapsed, 4)}
    if result.status in ("timeout", "candidate_exit", "child_error") and not result.checks:
        return 0.0, result.status, info
    if result.parse_ok is False:
        return 0.0, "applied_syntax_invalid", info
    if result.structure_ok is False:
        return 0.0, "structure_violation", info
    if result.sourced_ok is False:
        return 0.0, "source_error", info
    passed: dict[str, int] = {}
    ran: dict[str, int] = {}
    for category, ok in result.checks:
        ran[category] = ran.get(category, 0) + 1
        passed[category] = passed.get(category, 0) + int(ok)
    fractions = {c: min(passed.get(c, 0), n) / n for c, n in task.expected_categories.items() if n > 0}
    behaviour = sum(fractions.values()) / len(fractions) if fractions else 0.0
    complete = all(ran.get(c, 0) == n and passed.get(c, 0) == n for c, n in task.expected_categories.items()) \
        and set(ran) <= set(task.expected_categories) and result.error_class is None
    info.update(behaviour=round(behaviour, 6), all_pass=complete, checks_passed=sum(passed.values()),
                checks_expected=sum(task.expected_categories.values()), checks_ran=sum(ran.values()),
                by_category={c: [passed.get(c, 0), n] for c, n in task.expected_categories.items()})
    reward = config.w_behaviour * behaviour + config.w_all * float(complete)
    outcome = "all_pass" if complete else ("partial" if behaviour > 0 else "fail")
    return reward, outcome, info


class ExecReward:
    """``reward(ids, completions)`` over exec-env tasks; thread-safe through the harness."""

    def __init__(self, tasks: Mapping[str, ExecTask], harness: ExecHarness,
                 decoder: Callable[[Sequence[int]], str] | None, config: ExecRewardConfig | None = None) -> None:
        self.tasks = tasks
        self.harness = harness
        self.decoder = decoder
        self.config = config or ExecRewardConfig()

    def __call__(self, ids: Sequence[str], completions: Sequence[tuple[int, ...]]) -> list[dict]:
        return self.score([(i, tuple(c), None) for i, c in zip(ids, completions, strict=True)])

    def score_texts(self, ids: Sequence[str], texts: Sequence[str], stops: Sequence[str] | None = None) -> list[dict]:
        """Score decoded completions (``stop`` is ``eos`` or ``length`` per text)."""
        stops = list(stops) if stops is not None else ["eos"] * len(texts)
        return self.score([(i, None, (t, s)) for i, t, s in zip(ids, texts, stops, strict=True)])

    def score(self, samples: Sequence[tuple[str, tuple[int, ...] | None, tuple[str, str] | None]]) -> list[dict]:
        from sepalith.training.rl.reward_v21 import _check_ids

        records: list[dict] = []
        pending: list[tuple[int, ExecTask, tuple[str, ...]]] = []
        jobs: list[ExecJob] = []
        for index, (task_id, completion, text_stop) in enumerate(samples):
            task = self.tasks[task_id]
            record: dict = {"schema": RECORD_SCHEMA, "reward_policy": POLICY_ID, "row_id": task_id,
                            "family": task.family, "protocol_valid": False, "failure": None, "cap_hit": False,
                            "exact": False, "exact_match": False, "reward": 0.0, "outcome": None}
            records.append(record)
            failure: str | None = None
            text: str | None = None
            if completion is not None:
                record["generated_tokens"] = len(completion)
                if any(type(t) is not int or not 0 <= t < VOCAB_SIZE for t in completion):
                    failure, cap = "invalid_token_ids", False
                else:
                    failure, cap = _check_ids(list(completion), self.config.max_completion_tokens)
                if failure is None:
                    if self.decoder is None:
                        raise ValueError("token IDs need a decoder")
                    text = self.decoder(list(completion[:-1]))
            else:
                assert text_stop is not None
                text, stop = text_stop
                cap = stop == "length"
                failure = "missing_canonical_eos" if cap else None
            record["cap_hit"] = cap
            body: tuple[str, ...] = ()
            if failure is None and text is not None:
                try:
                    parsed = parse_output(text, task.context)
                except (ProtocolError, TypeError, ValueError) as error:
                    failure = f"parse_error:{error}"
                else:
                    if parsed.status != "accepted" or parsed.operation is None:
                        failure = parsed.reason or "invalid_protocol_output"
                    else:
                        record.update(protocol_valid=True, operation=parsed.operation)
                        body = tuple(parsed.body)
            if failure is not None:
                record.update(failure=failure, reward=REWARD_CAP_HIT if cap else REWARD_PROTOCOL_INVALID,
                              outcome="cap_hit" if cap else "protocol_invalid")
                continue
            if record["operation"] == "no_op":
                record.update(outcome="missed_edit", reward=0.0)
                continue
            record["exact_match"] = normalize_lines(body) == normalize_lines(task.target_body_text.split("\n"))
            bad = forbidden_constructs(body, task.gold_text)
            if task.forbid_in_region and re.search(r"(?<![A-Za-z0-9._$@])" + re.escape(task.forbid_in_region)
                                                   + r"(?![A-Za-z0-9._(])", _code_only("\n".join(body))):
                bad.append(f"stale_name:{task.forbid_in_region}")
            if bad:
                record.update(outcome="forbidden_construct", reward=0.0, forbidden=bad)
                continue
            pending.append((index, task, body))
            jobs.append(ExecJob(task.splice(body), task.suite, task.expect, task.timeout, f"{task_id}:{index}"))
        for (index, task, body_lines), result in zip(pending, self.harness.run_many(jobs), strict=True):
            record = records[index]
            reward, outcome, info = score_result(task, result, self.config)
            if self.config.exact_bonus and record["exact_match"]:
                reward += self.config.exact_bonus
            lines = sum(1 for line in body_lines if line.strip())
            gold_lines = sum(1 for line in task.target_body_text.split("\n") if line.strip())
            if self.config.overlong_scale < 1 and lines > max(self.config.overlong_factor * gold_lines,
                                                               gold_lines + self.config.overlong_slack):
                reward *= self.config.overlong_scale
                record["overlong"] = {"lines": lines, "gold_lines": gold_lines}
            record.update(reward=round(min(1.0, reward), 6), outcome=outcome, exact=bool(info.get("all_pass")),
                          execution=info)
        return records


def make_exec_reward(tasks_path: Path, *, workers: int = 8, tokenizer_json: Path | None = None,
                     config: ExecRewardConfig | None = None) -> ExecReward:
    """Build an ``ExecReward`` (callers own ``reward.harness.close()``)."""
    decoder = None
    if tokenizer_json is not None:
        from sepalith.training.rl.reward_v21 import tokenizer_decoder
        decoder = tokenizer_decoder(tokenizer_json)
    return ExecReward(load_tasks(tasks_path), ExecHarness(workers=workers), decoder, config)


class RoutedReward:
    """Dispatch each prompt to the reward that owns it: exec-env tasks to ``ExecReward``, legacy
    TRAIN-row replay to reward v2.1 (``training.grpo.trainer.PoolReward``). Same call signature."""

    def __init__(self, exec_reward: ExecReward, fallback: Callable[[Sequence[str], Sequence[tuple[int, ...]]], list[dict]]) -> None:
        self.exec_reward = exec_reward
        self.fallback = fallback

    def __call__(self, ids: Sequence[str], completions: Sequence[tuple[int, ...]]) -> list[dict]:
        out: list[dict | None] = [None] * len(ids)
        for owner, picks in ((self.exec_reward, [i for i, x in enumerate(ids) if x in self.exec_reward.tasks]),
                             (self.fallback, [i for i, x in enumerate(ids) if x not in self.exec_reward.tasks])):
            if picks:
                for index, record in zip(picks, owner([ids[i] for i in picks], [completions[i] for i in picks]), strict=True):
                    out[index] = record
        assert all(record is not None for record in out)
        return out  # type: ignore[return-value]
