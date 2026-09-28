# sm120 varlen dispatch probe v2

This fresh packet corrects the evidence boundary in the v1 audit probe. The xFormers dispatcher names and Torch implementation-expected names remain labeled as inferred or implementation-declared. A separate untimed `torch.profiler` pass uses CPU activity only and records the operator keys emitted by each real forward/backward invocation. If profiling is unavailable, the report leaves every actual-key list empty and retains inferred names in a separate field.

The numerical comparison retains the scalar all-position loss and Q/K/V gradients. It adds a deterministic BF16 random cotangent and separately compares Q/K/V gradients, reducing cancellation from a uniform scalar cotangent. For each multi-document profile it perturbs Q/K/V inside one document, reruns each backend, requires exact zero output change outside that document, and requires a nonzero change inside it. The single-document profile records that isolation is not applicable.

Profiles remain GQA 16 query heads, 2 KV heads, dimension 128, with exact lengths 7,260, 16,040, and 16,384. Timing remains alternating order with two warmups and five synchronized measurements per backend. Profiler work is outside those measurements. The maximum runtime remains 600 seconds, model loading and training are forbidden, and only root may admit and launch the GPU probe.
