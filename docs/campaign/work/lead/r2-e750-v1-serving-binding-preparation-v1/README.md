# E750 V1 serving binding preparation

This packet binds the current E750 Q8 development candidate, PRM-03 tokenizer/renderer, CUDA b10453 runtime, and the isolated b4 rollback profile. It is a preparation artifact. It authorizes no model load, editor launch, benchmark, final-set access, promotion, or release.

The accepted VSIX and E750 native evaluator both bind `contextSize=4096` and `maxOutputTokens=192`. The 192-token DEV run hit the cap on five of 75 cases (three finish-block and two no-op). Therefore 192 remains the reproducible baseline, while 384 and 768 are explicit development candidates. The latter two need fresh reviewed VSIX/manifest and evaluator/controller overlays; editing the frozen accepted sources would invalidate their identities.

The cap experiment must keep all 75 development case IDs and exact 43 edit/32 no-op denominators. It chooses the smallest cap that removes length stops without reducing exact/no-op/protocol performance or prompt coverage. Every request retains the realistic five-second deadline. Longer diagnostic timeouts are reported separately and cannot count as interactive success.

Serving latency uses the reviewed v5 paired harness. Ordinary E750 remains acceptable when every speculative arm fails output parity, the 1.4x point-speed threshold, the paired trace-cluster confidence lower bound, or p95 latency. The target-host editor check must use `/tokenize` followed by integer-ID `/completion`; the legacy `/v1/completions` route is not equivalent.

The notebook was inspected read-only. Code 1.137.0, Xvfb, R, and tmux are present. No llama-server, editor-development, Xvfb, or campaign process was observed, and the checked ports were free. The b4 model/runtime/VSIX paths are present; runtime and VSIX hashes match. The E750 Q8 target path is absent, so staging and a full target-host hash remain required. No service or benchmark was started.
