# DAT-10 independent replay sample

`replay_sample.py` performed an independent source/provenance replay over 20
TRAIN rows from eight source-walk shards and eight packages. It reopened and
hashed the raw scenario files, selected raw lines, normalized R files,
DESCRIPTION files, global split registry, and protected CPT partition. It
also invokes the immutable strict protocol validator at SHA `7f042596...`.

The sample contains 13 roxygen rows and seven no-ops. Six no-ops pass the
independent provenance and family-specific unchanged-source predicate. All 13
roxygen rows pass provenance and remain queued for the full semantic analyzer.
One `blank_between` no-op is held because its reconstructed source window has
zero exact occurrences in the normalized file. No producer boolean was used
as evidence.

No R code was executed. The tree-sitter parser only parsed source bytes. No
source, prompt, or target text is written to the output ledger. This packet
does not admit data or launch the full 41-shard replay.
