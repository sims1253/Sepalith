# Full-weight CPT long-context resource fixtures v1

This packet provides two exact full-length all-token CPT rows at 8,192 tokens and two at 16,384 tokens for root-owned resource probes. The rows come from two complete, committed TRAIN documents: one 37,154-token `geospatialsuite` document and one 33,063-token `binGroup` document. The generator reads only those two committed group artifacts.

The copied accepted `r2-cpt-lossless-rechunk-v2` closure validates and reassembles all 36 original 2K chunks before rechunking. Each context output conserves all 70,217 source payload tokens exactly once and has one supervised terminal EOS per document. The smoke fixtures select continuation chunk 1 from each document. Every selected sequence exactly fills its context, contains one masked prior-token carry, supervises every new raw-code token, masks the nonterminal EOS, and contains no padding or truncation.

`materialized/fixture-manifest.json` binds the committed group receipts, source rows, document inventories, copied rechunker, root smoke harness, selected row hashes, ranges, label counts, and conservation proof. The full rechunk output remains available so the selected rows can be replayed against their complete-document lineage.

`root-commands.json` contains preparation-only command arrays for the existing full-weight smoke harness. Root owns the model decision, CUDA lease, resource guard, reports, and execution. These fixtures provide memory and throughput evidence only; they do not authorize training or establish a scientific context choice.
