# Source-schema integration audit

The A100 and varlen v1 packets remain unchanged. Both used custom source-manifest schemas, but neither current front door verified that closure. Wiring the copied stage-transition trainer directly into A100 would also reproduce the reported schema rejection.

Fresh A100 v2 and varlen v2 packets add a strict runtime closure verifier. It checks the expected packet-specific schema, unique confined paths, file sizes, hashes, regular files, and symlink rejection. A100 v2 also closes source-recipe and checkpoint-metadata pin gaps. Varlen v2 records the verified closure in its terminal report for independent verification.

The audit tests invoke the unchanged stage trainer against the real A100 manifest and observe the expected rejection. They then invoke the actual A100 v2 preflight and varlen v2 verifier with their real manifests. No GPU process ran.
