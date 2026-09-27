# DAT-10 semantic queue: remaining committed shards 12+

This packet reuses the byte-pinned v3 semantic worker and owns one fresh bounded CPU run over replay shards 12–26. The original independent replay was reobserved as historical and exited 0; it was not restarted. Shards 0–11 are prior semantic scope. Intake preflight validates the v3 replay index, every selected provenance receipt and ledger, exact queued-ID disjointness against prior semantic outputs, and fresh output state. The queue never receives a terminal provenance manifest, so its result remains partial review only.

Bulk output is on E. Results are review evidence only; root admission must still verify source/license/global split, protected/heldout exclusion, exact prompt/target dedup, geometry, and terminal protocol.
