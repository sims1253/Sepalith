from pathlib import Path
import json,re,hashlib
root=Path(__file__).resolve().parents[1]
extension=(root/'source/extension/src/extension.ts').read_text()
package=json.loads((root/'source/extension/package.json').read_text())
route=(root/'source/extension/src/expanded_provider_route.ts').read_text()
assert package['contributes']['configuration']['properties']['sepalith.expandedProvider']['default'] is False
assert 'if (configured.expandedProvider)' in extension
assert 'else {\n      context = selectPromptContext' in extension
assert 'requestStartedAt + c.requestTimeoutMs' in extension
assert 'signal: lease.controller.signal' in extension
assert 'generationReserve: request.client.maxOutputTokens' in route
assert 'providerSelector(request.client)' in route
assert 'currentNamespacePathIdentity' in route
assert 'Math.min(2000, remainingMs)' in route
for name,expected in [('namespace_evidence.R','97a56820bb237b4d6e40216f932b2514a306985c0b67cbb3afe6b91b22bf1847'),('source_imports.R','cb0822dd3d9d50018ec20c2992b017506d232ca56169570d0751256e69456250')]:
    p=root/'source/extension/resources/expanded-provider'/name
    assert hashlib.sha256(p.read_bytes()).hexdigest()==expected
print(json.dumps({'status':'PASS','opt_in_default':False,'existing_primary_else_preserved':True,'shared_request_signal':True,'profile_context_and_output_cap_source':'NativeCampaignClient','resource_helper_pins':2},sort_keys=True))
