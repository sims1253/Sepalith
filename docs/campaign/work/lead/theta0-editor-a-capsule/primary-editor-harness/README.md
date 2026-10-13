# Primary managed editor harness

`extension.js` runs inside an actual VS Code extension host. It reads the
application-scoped `sepalith` settings that the launcher wrote below the
isolated `user-data/User/settings.json`, then creates two synthetic `.R`
buffers in the disposable workspace and records editor events, the
identity-bound prompt copied by the accepted extension command, a document
integrity smoke after a concurrent edit, and the command surfaces available
for inline or multiline acceptance. It also calls the real managed
`/tokenize` and integer-ID `/completion` endpoints after `sepalith.startServer`.

The harness never supplies a manual `serverPath`. A successful run therefore
requires the launcher to provide the validated HTTPS primary manifest, the
exact selected theta0 step1000 Q8_0 model override, and the managed Vulkan bundle. The direct native
probe reuses the exact prompt captured from the active PRM-03 editor command;
it does not reconstruct an approximate prompt. The result is mechanical route
evidence; it does not promote a model or claim suggestion quality. If a GUI,
clipboard, native route, or acceptance surface is absent, the result records
a bounded failure or pending check. Inline ghost nonpublication and request
cancellation remain pending because this harness cannot observe the private
provider transport or VS Code ghost buffer.
