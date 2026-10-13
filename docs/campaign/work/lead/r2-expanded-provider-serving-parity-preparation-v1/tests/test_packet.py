#!/usr/bin/env python3
import hashlib,json
from pathlib import Path
P=Path(__file__).resolve().parents[1]
PLAN=P.parents[4]
def h(p):return hashlib.sha256(p.read_bytes()).hexdigest()
active=PLAN/'docs/campaign/work/lead/r2-e750-output-cap-overlay-v1/source/extension/src'
provider=PLAN/'docs/campaign/work/lead/r2-noop4100-parse-retry-preparation-v1/source'
assert h(active/'campaign_protocol.ts')==h(provider/'campaign_protocol.ts')=='ec4ab1529cb7ade10be28343600f26be027f24239009ec354a357a7caff4b824'
assert h(active/'context_select.ts')=='a72edaf733395d7b4839f2ca9ed0c2092e4051359279f6839c93e2d5d04b9d59'
assert h(provider/'context_select.ts')=='413f0a1c2fab480fcf36c3fdd221a03a359a05d52dec81475193cd6fc1870576'
assert h(P/'source/provider/context_select.ts')==h(provider/'context_select.ts')
assert h(P/'source/provider/full_document_policy.ts')=='dd671fa12a839d00e6446bd07d36c02e33518290b6e95faf8187163aa08f0a95'
assert h(P/'source/provider/namespace_runtime.ts')=='47c7109eb5d81968cbbb69c1a6b694fd73be6241cc93b3fbae39005cc4897817'
rows=[]
for name in ('e677-preedit.jsonl','valid-control.jsonl'):
 row=json.loads((P/'inputs'/name).read_text());assert hashlib.sha256(row['preedit_text'].encode()).hexdigest()==row['preedit_sha256'];rows.append(row['row_id'])
assert rows==['e677ee6a8da38436f4bdb6b6','43b24d15b32aae89fdf72245']
text=(P/'source/expanded_primary_context.ts').read_text()
for needle in ('stillCurrent','deadlineMs','AbortSignal','namespacePathIdentity','providerSelector'):
 assert needle in text
assert 'sourceImportEvidence' in (P/'source/provider/namespace_runtime.ts').read_text()
print('PASS packet exact active/provider hashes and two TRAIN fixture identities')
