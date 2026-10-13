# Three geometry holds v2

This revision closes the provider-contract gap found in v1. It reconstructs the exact full raw pre-edit document and global cursor for each row, then invokes the real TypeScript primary path. Two reconstructed pre-edit documents retain mixed LF/CRLF separators, which the primary path rejects before prompt construction. The third pre-edit is uniform CRLF, but its captured target block has an internal lone LF; the actual protocol maps the wire target uniformly to CRLF and therefore does not reconstruct the pinned source bytes.

Normalizing each entire document to LF would make the primary path accept it, but that changes non-target source bytes and violates this task's line-ending preservation requirement. The mixed target block in row `4fc746562d6b6f36c95df72b` also cannot be reproduced from `target_lines` plus the protocol's single `document_eol` value.

The correct outcome is three named protocol holds, zero provider predictions, and no training admission. V1's short candidate windows remain useful audit evidence only and are superseded for provider readiness by this packet.
