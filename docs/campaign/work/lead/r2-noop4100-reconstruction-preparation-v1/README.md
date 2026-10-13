# No-op 4100 reconstruction preparation

This fresh packet joins the full 41-shard, 4,227-row no-op census with the
frozen v4 source-backed recovery ledger.  It emits 4,100 target-free provider
inputs and 127 named holds: 121 provenance holds and six mixed-EOL geometry
holds.  It does not run the provider or admit training data.

The reconstruction keeps all duplicate source geometries.  Deduplication is
deferred until actual rendered prompt/target identities exist.  Missing or
unreadable sources, source hash changes, and input-pin changes terminate the
whole command; they are not converted into row holds.

`source/prepare_noop4100.py` recomputes each recovered geometry from the raw
source.  It projects the recorded replacement start into a global UTF-16
cursor, including a source-window prefix column when the window begins
mid-line.  Uniform CRLF is normalized only for matching and position math;
the emitted preedit remains byte-exact raw source.

The full command in `commands.json` is prepared for root execution into a
fresh E-drive directory.  Provider 16K/32K lane preparation remains a separate
manifest-dependent follow-up after this reconstruction has completed and its
4,100/127 terminal manifest has been independently reviewed.
