#!/usr/bin/env bash
set -euo pipefail
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1
python3 "$(dirname "$0")/repair_geometry.py" \
  --semantic-manifest /mnt/e/sepalith/campaign-20260915/data-work/Sourcewalk-semantic-streaming-queue-shards6plus-v1/shard-0011/manifest.json \
  --expected-semantic-manifest-sha256 fa95730378bf9eccd1ab797ee9101158eb91eb6dc9c1f4749a34ed7517ef1a14 \
  --semantic-ledger /mnt/e/sepalith/campaign-20260915/data-work/Sourcewalk-semantic-streaming-queue-shards6plus-v1/shard-0011/semantic-ledger.jsonl \
  --expected-semantic-ledger-sha256 34609322af75cfe4d373273bcf38833691d35a91b11911602e4bd93d604a50d5 \
  --provenance-ledger /mnt/e/sepalith/campaign-20260915/data-work/Sourcewalk-independent-replay-v3/full-01/shards/shard-0011/ledger.jsonl \
  --expected-provenance-ledger-sha256 acb8167ca0a2e116887132f44de820ed2d8e52fcdd90305cbd12ec4201425b7d \
  --candidate-packets /mnt/e/sepalith/campaign-20260915/data-work/DAT10-novel-v1/source-walk-shards-v1/shard-0011/structured-materialization-v1/candidate-packets.jsonl \
  --expected-candidate-packets-sha256 662a5bffb52d8817efa29a62fce96dcee0b0d584f9bb3daa91a867cf12882943 \
  --holds /mnt/e/sepalith/campaign-20260915/data-work/Semantic4554-root-preparation-v1/shard-0011/preparation-holds.jsonl \
  --expected-holds-sha256 792b4d23809ebe959d5f4602dbe281ec64adddee9276249bbdb23f157a027728 \
  --output /mnt/e/sepalith/campaign-20260915/data-work/Semantic-three-geometry-holds-v1/review-01
