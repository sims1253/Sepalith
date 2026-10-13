# RUN-06 notebook editor-mechanics preparation

This packet prepares one root-admitted, disposable VS Code/Xvfb run on the notebook. It exercises the frozen future expanded-provider bundle through its real `extension.ts -> ExpandedProviderRoute -> NativeCampaignClient -> inline suggestion/application` path. The managed localhost sidecar is a scripted mock. The run therefore tests editor mechanics, protocol rejection, cancellation and identity invalidation; it supplies no model-quality or production-latency evidence.

The candidate bundle is copied byte-for-byte from `r2-expanded-provider-extension-integration-preparation-v1` (source-manifest SHA `e2bad4...`, bundle SHA `336d03...`). Its contributed setting remains `expandedProvider=false`. Only the disposable profile sets it true. The user's installed extension, b4/E750 defaults, documents and profile are never read or changed.

Two fixed TRAIN pre-edit buffers are included with their source hashes, DESCRIPTION and NAMESPACE. No DEV or final content and no expected target text is present. The launcher makes per-run copies under the private workspace. Its mock serves canonical edit, canonical no-op, malformed and delayed responses on loopback only. The mock uses a valid primary manifest and exact runtime/profile constants, while every asset and the dummy model state clearly that they are mechanics-only.

Root must fill an admission copied from `admission.template.json`, stage the prepared tar into a fresh remote prefix, and use the command in `commands.json`. The outer 240-second timeout and inner 220-second bound kill only the recorded detached Code process group. The run retains logs and evidence and verifies the dedicated port is free after cleanup. No run has been launched by this preparation.
