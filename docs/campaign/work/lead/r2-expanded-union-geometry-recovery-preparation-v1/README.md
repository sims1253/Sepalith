# Expanded union geometry recovery preparation

This target-independent mapper derives canonical application geometry from exact `PromptContext` replacement ranges and selected context windows. It uses the replacement start as the document-global LSP cursor; the auxiliary region-local cursor is never substituted. Full snapshots are reconstructed with their declared LF/CRLF convention and must match `content_sha256`. Clipped or otherwise non-verifiable buffers retain authoritative context geometry but explicitly report that full-buffer verification was unavailable.

Prepared inputs cover original 15,006 and the 10,682 admitted candidates selected from the 10,948 semantic provider inputs. The other 5,185 existing semantic candidates still need root-bound selected-context paths; their current provenance records do not contain cursor/range geometry. They remain unresolved and are not excluded. No token target or label participates in geometry selection.
