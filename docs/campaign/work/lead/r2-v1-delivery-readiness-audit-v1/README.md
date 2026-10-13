# V1 delivery-readiness audit

This is a read-only readiness audit. It does not select a release, open sealed
final data, launch a server, or close the longer full-weight CPT campaign.

The current artifacts support assembling a **review package** for a near-term
V1 independently of the ongoing full-weight CPT run:

* Primary candidate for root review: E750 Q8, with the pinned PRM03 renderer,
  tokenizer, and CUDA llama.cpp bundle.
* Rollback: b4 Q8 with its matched legacy renderer VSIX and CPU llama.cpp
  runtime.

That package is not yet a release package. E750 has current native DEV quality
evidence but has not run through the real editor route. The editor evidence in
the campaign applies to selected500/theta0 or b4, not E750. The final target
serving packet is also unbound, and REL-01 through REL-10 remain open.

See `audit-result.json` for paths, hashes, current observations, evidence
limits, and the shortest safe route to a V1.
