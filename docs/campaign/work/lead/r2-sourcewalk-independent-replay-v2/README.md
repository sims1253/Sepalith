# DAT-10 independent replay v2

This packet implements the full 76,279-row provenance replay without admitting
any row for training. The denominator is 72,052 mechanically unique roxygen
rows plus all 4,227 converted no-ops. Seven no-ops already named by the
mechanical hold ledger remain in the denominator and carry their hold reasons.

`full_replay.py build-index` verifies all 41 token-row receipts, the exact hold
ledger, and the 76,279-row denominator. It then scans each raw scenario JSONL
once, gates on stable file identity and whole-file SHA, validates exact source
line/family/package joins, and atomically publishes a per-shard raw index.

`full_replay.py replay` processes one shard at a time. It independently reopens
normalized source and DESCRIPTION files with before/after stat gates, parses the
DESCRIPTION, and requires a positive decision from the pinned campaign license
parser. Empty, malformed, restricted, non-FOSS, and unknown licenses fail. It
also invokes the pinned strict token validator, validates source-backed no-op
geometry, and commits the shard with an immutable receipt. Existing shards are
reused only when all bindings and output bytes still match.

Roxygen rows that pass provenance remain explicitly queued for semantic
analysis. No empty analyzer state can pass, and no row is admitted here.

The bounded sample reopens 20 TRAIN rows across eight packages and eight
shards. All 20 pass the tightened provenance gates; 13 roxygen rows remain in
the semantic queue and seven no-ops remain candidates for root review. Row
`824a351dc66d530f239aa3d9` was a harness false hold: its upstream file uses
uniform CRLF while the builder records LF. The exact content window occurs once
after only CRLF-to-LF normalization; no other whitespace normalization is
allowed.

The full command is in `run_full.sh`. It is prepared for root review and has
not been launched.
