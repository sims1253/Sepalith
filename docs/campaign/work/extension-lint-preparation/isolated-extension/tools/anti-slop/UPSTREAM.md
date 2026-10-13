# Vendored anti-slop

Source: https://github.com/dmmulroy/anti-slop
Revision: e8c4880471b23ab7f216fba7b27d173a6ef07d4c
License: MIT (see LICENSE)

Copied the generic entry point, rules, shared helpers, and rule tests from src/.
The Effect plugin is omitted. Generic source and tests are unchanged. Added a
local package.json to mark these TypeScript files as ES modules.

Project rule choices live in ../../.oxlintrc.json. See the repository's
docs/CODE-QUALITY.md for the policy and upgrade procedure.
