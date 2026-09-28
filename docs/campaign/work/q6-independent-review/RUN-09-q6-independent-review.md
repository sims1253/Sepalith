# RUN-09 Q6 independent DEV review

The Q6 quality server completed all 75 pinned DEV cases. The remote server and local client both exited successfully, and the independent CPU reclassification reproduced every stored protocol, exact-region, no-op, cap, and denominator field.

The evaluated artifact is Q6_K, SHA256 `b11ffcc093b78261af1c5eb450feefcbca6ee35c0cecc22133236506410e143e`, 2,199,741,184 bytes. The remote launch bound that exact model path to the b10453 Vulkan server with build receipt SHA `6f4b5c3064e74cd25013e142417aa6cc71aa66a7591d442a978d74f5a75ba7ed`, context 4096, six threads, batch/ubatch 256, and no profiling. The remote device audit recorded the Vulkan library and `/dev/dri/renderD128`; the server log records `offloaded 43/43 layers to GPU` and Q6_K file type. This verifies the requested runtime/backend identity for the quality run.

The panel was verified byte-for-byte against DEV SHA `b16c5892635d6e9a5e9e789677057c85a5e875c907c163e91d39f25bc6ad3d21`, with 75 rows, 43 edit cases, 32 strict no-op cases, and case-ID order SHA `6e0c4c5bd7de4f0d9aa6eab602796c13da110f56edab6b0dd8a9e04fb66ee6bb`. The reclassifier loads the pinned `campaign_eval.py` SHA `7064385d63e3900b87241da75525897acb6b55d9ce1a023689c9addfbaacfc9f` and `campaign_protocol.py` SHA `5a869329be74d8177769901bc771aa3dc6e19dad1f2d97fca149e5c38f840156`. It parses each stored output against the panel's exact `PromptContext`, independently verifies the raw-output and returned-ID hashes, and does not contact the server or load a model.

Independent Q6 counts are 26/43 exact edit regions, 20/32 correct strict no-ops, 11 strict-no-op false positives, 70/75 protocol-valid responses, 5 cap hits, 5 mechanical/protocol failures, 46 exact regions overall, and 0 transport failures. All five mechanical failures are also cap hits: three `finish_block`, one `no_op`, and one `roxygen_drafting`. The other families are protocol-valid with exact edit counts of format 6/7, na_rm 5/6, pipe 8/8, and rename 7/8. The no-op family has 20/32 strict correct and 11 false-positive suggestions.

| Candidate | Exact edits | Correct no-ops | No-op false positives | Protocol-valid | Cap hits |
| --- | ---: | ---: | ---: | ---: | ---: |
| Q8 native | 26/43 | 19/32 | 11 | 70/75 | 5 |
| F16 | 25/43 | 19/32 | 12 | 70/75 | 5 |
| Q4_K_M | 26/43 | 18/32 | 10 | 68/75 | 7 |
| Q6_K | 26/43 | 20/32 | 11 | 70/75 | 5 |

Relative to the supplied Q8 summary, Q6 has the same edit exactness, false-positive count, protocol denominator, and cap count, with one additional correct strict no-op. The Q6 artifact remains an intermediate step-500 candidate. The quality artifact reports native logits/NLL as unavailable, and this panel is a protocol/region/no-op diagnostic; these results do not establish R semantic validity, latency superiority, or promotion. Root's separate 5-second latency/cancellation comparison remains required.

