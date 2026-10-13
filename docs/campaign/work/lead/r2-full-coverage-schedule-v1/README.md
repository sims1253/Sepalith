# Full-coverage SFT schedule preparation

`audit_e_schedule.py` independently verifies the actual E schedule. E draws all 11,505 admitted IDs by update 868. Its 16,000 draws contain 12,000 edits and 4,000 no-ops. All complete target bodies, protocol terminals, and EOS labels are present; 1,963 unique targets exceed the old DEV limit of 192, while the largest supervised target is 933 tokens.

`full_coverage_schedule.py` derives its horizon from the input pool. It validates the complete token geometry, groups rows by actual 512/1024/2048/4096 sequence lengths, exhausts every unique edit and no-op before the corresponding pool can replay, and names every replay. It has no target-length cap. A target longer than 1,024 remains eligible when the complete sequence fits the selected context. Rows beyond the selected context are listed explicitly as excluded rather than silently dropped.

The concrete admitted 15,008-row schedule uses 1,160 updates and 18,560 draws. Every batch has 12 edits and 4 no-ops. It covers 13,914 unique edits and 1,094 unique no-ops, then uses six edit batch-completion alignment replays and 3,546 no-op ratio/length alignment replays. The schedule is preparation only and binds no parent, recipe, optimizer, or launch.

Rebuild any later admitted pool with:

```text
/home/m0hawk/Documents/Sepalith/.venv-sft/bin/python -B full_coverage_schedule.py --rows ABSOLUTE_TOKEN_ROWS --rows-sha256 EXACT_SHA256 --output FRESH_OUTPUT.json --effective-batch 16 --noop-per-batch 4 --max-sequence 4096 --seed 3407
```

Prospective roxygen additions must first be admitted into a new exact token-row artifact. The generator then recalculates the necessary updates and all ID/token denominators; it does not inherit 1,000 or 1,160 updates.
