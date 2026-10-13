# Card status files

Each card keeps one file here, named `<card>.json`, for example `C04.json`.
Update it in the same pull request as the card's changes. For night runs,
update it in the morning report pull request.

```json
{
  "card": "C04",
  "state": "in_progress",
  "updated": "2026-09-29T14:05:00+02:00",
  "branch": "plan/C04-sft-recipe-v2",
  "pull_request": "https://github.com/sims1253/Sepalith/pull/NN",
  "outputs": {
    "schedule_sha256": "...",
    "hf_paths": ["campaign-20260915/C04/..."]
  },
  "rule_decisions": [
    {"rule": "R1", "inputs": {"edit": 31, "valid": 70}, "outcome": "continue"}
  ],
  "question_for_user": null,
  "notes": "One or two sentences on what changed and what remains."
}
```

`state` is one of `not_started`, `in_progress`, `blocked`, `needs_user` or
`done`. With `needs_user`, put exactly one question in `question_for_user`.
Include the numbers needed to answer it and your recommended answer.
