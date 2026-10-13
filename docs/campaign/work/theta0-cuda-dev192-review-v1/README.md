# RUN-05/RUN-09 theta0 CUDA output-192 review

`review_dev192.py` is a read-only CPU review of the completed Q8_0 CUDA
profiles `b256` and `b1024`. It independently renders each corrected DEV
context, tokenizes it with the pinned HF tokenizer (`split_special_tokens=True`,
manual BOS 0), decodes every saved native response, and recomputes the native
EOS/cap, parser, protocol, exact-region, edit, and no-op decisions. It also
checks the saved source pins, model provenance string, props, GraphOpt0
environment, graph geometry, offload, and clean exit artifacts. The model
file is never opened or hashed.

Run it from the plan worktree with the pinned CPU environment:

```sh
/home/m0hawk/Documents/Sepalith/.venv-sft/bin/python \
  docs/campaign/work/theta0-cuda-dev192-review-v1/review_dev192.py \
  --out-dir docs/campaign/work/theta0-cuda-dev192-review-v1
```

The generated `review-report.json` retains all 75 row comparisons for the
current pair and audits the historical output-512 run under a separate
denominator. `input-manifest.json` records hashes for every source and saved
run artifact used by the review. The receipt is
`docs/campaign/receipts/RUN-09-theta0-cuda-dev192-review-v1.json`.

The independent results are:

| profile | protocol | cap hits | exact region | exact edits | correct no-op | no-op false positives |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Q8_0 b256, output 192 | 69 | 6 | 51 | 26 | 25 | 5 |
| Q8_0 b1024, output 192 | 70 | 5 | 50 | 26 | 24 | 6 |
| historical Q8_0 output 512 | 71 | 4 | 51 | 26 | 25 | 5 |

The current profiles differ in seven output rows. The matched no-op
`dat07-derived-d1c76ba29b30040a7561ce3e` is correct under b256 and becomes a
protocol-valid no-op false positive under b1024. The review therefore records
`hold` for b1024 promotion; the one recovered capped row does not clear this
matched-case regression.
