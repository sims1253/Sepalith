# Reports and raw evidence

Read the reports in Git to understand what we tried, what happened and what we
decided. Use the archive to inspect the underlying logs and process records.

```text
Git repository                         Private Hugging Face bucket
├── source and tests                   sepalith-raw/
├── experiment recipes                 └── sha256/<content hash>/
├── reports with results and limits        ├── training logs
└── docs/artifacts/index.json              ├── saved process observations
    └── paths, hashes and sizes           └── historical comms snapshots
```

The bucket is [scholzmx/sepalith-raw](https://huggingface.co/buckets/scholzmx/sepalith-raw).
It is private. A public clone includes the summaries and index; fetching raw
objects requires account access. Keep test fixtures and input specifications in
Git so ordinary development does not require bucket access.

Hugging Face buckets have no version history. We put each object under its SHA256
hash and check downloaded bytes against the Git index. Never replace an object
with different content at the same path. This convention detects changes; it
does not provide storage-enforced immutability. See the
[Hugging Face bucket documentation](https://huggingface.co/docs/hub/storage-buckets).

## First archive

The [index](artifacts/index.json) identifies three files from commit
`6b597058047aebbb70669e5042c4607be76898a0`: a historical comms board, a saved
masked-diffusion training log, and the latent-memory preparation's process
observation. All three were uploaded, downloaded and checked byte for byte.
The total is 337,700 bytes.

The training log records execution telemetry. Its research conclusion belongs in
[the POC-DIFF results](../experiments/training/poc_diff/RESULTS.md).
The process observation records that the machine was occupied when latent-memory
preparation finished. The [latent-memory results](../experiments/latent_memory/RESULTS.md)
state that its real learnability experiment has not run.

The historical board snapshot preserves decisions and handoffs. The active
`comms/board.md`, `comms/gpu.md` and queue remain available locally and tracked
while the legacy experiment drain is pending. The archive is not the live
coordination channel. A later protocol change can separate local live messages
from the durable decision summaries in Git.

## Retrieve evidence

Use an authenticated Hugging Face environment with `huggingface_hub==1.13.0`
(the version used for this archive):

```sh
python scripts/artifacts/fetch.py experiments/latent_memory/resource-observation.json
```

The command prints the verified local cache path. `--output /absolute/path`
selects a destination. It refuses to overwrite different bytes. Credentials stay
in the environment or Hugging Face credential store.

## Writing a report

State the question, result and decision in plain prose. Include enough numbers
to judge the conclusion, then explain uncertainty and what the evidence cannot
establish. Link the recipe and indexed raw evidence. Failed experiments still
get a report. Successful commands do not establish scientific success.

Apply orwell-writing and humanizer to the prose. Use show-me when a small table,
plot or diagram explains the result better than another paragraph. Keep raw
console output and session narration in the archive.

Archive a closed file before removing it from tracking: inspect its contents,
record its origin and hash, upload it privately, download it, verify it, and
update its readers and links. Keep local copies during the transition. Never
sync the whole checkout or credential directories into the bucket.
