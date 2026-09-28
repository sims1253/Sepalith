# RUN-04 editor TLS diagnosis

Observed at `2026-09-12T23:32:05+02:00` for the terminal notebook run
`primary-editor-managed-b`. This is a read-only diagnosis; the worker did not
launch an editor/server, change settings, read the private key, or make an HTTP
request.

## Observed run

The existing run used VS Code 1.137.0 on the notebook and an HTTPS loopback
manifest:

* VS Code commit: `645f29cc3176500b4b5762ba887cf2a7f0ffdf2c`
* Node: `v26.8.1`
* manifest URL: `https://127.0.0.1:40877/manifest.json`
* process CA: `runs/primary-editor-managed-b/manifest-ca.pem`
* VSIX: `dc45d0d843f479c8ba3f4814f4eb4dd4ed71e5bc75a776fc04c7f30e5987623f`
* manifest request audit: one `GET /manifest.json`, status `200`
* host result: `primary_managed_route_failed`, `native readiness deadline exceeded: fetch failed`

The parent separately tested the same endpoint while the smoke server was
alive with the process-scoped `NODE_EXTRA_CA_CERTS` and got HTTP 200 with 4241
bytes. The run's `manifest-ca.pem` was read only for its SHA256 and certificate
fingerprint; `manifest-key.pem` was not read.

## Source evidence

The pinned Sepalith source is unchanged. `runtime.ts` (`sha256
4de415237088984d53eba6b182dba975c12216a516c52f9fecb3a916dc841616`) calls the
ambient global `fetch(url, { signal })` at lines 419--420 in `loadManifest`.
There is no custom CA, dispatcher, `https.request`, or proxy option in that
call.

The installed extension host is
`/usr/share/code/resources/app/out/vs/workbench/api/node/extensionHostProcess.js`
(`sha256
110404a6c132243d65b45cfd71bb426229291765828190c8d5bf28873ce2a90d`,
2,252,012 bytes). Its byte-offset 2,104,950 code replaces
`globalThis.fetch` and routes normal requests through
`@vscode/proxy-agent`'s `createFetchPatch`; the `electronFetch` branch is
disabled by default (`x9 = false`) and is disabled for a remote extension host.
The extension host's environment denylist removes `NODE_OPTIONS` and
`VSCODE_NODE_OPTIONS`, but does not list `NODE_EXTRA_CA_CERTS` (offset
1,450,172).

The archived proxy-agent is version `0.45.0`, source SHA256
`22ac42e09faba49374a93b5167cddff7adaeb4108c6e240ed2389566cfa594e9`, from
`node_modules.asar` SHA256
`746f495f05cd5706914fb8fc6e4295585903bcd812aa51727a28cf0e576a280c`.
Its extracted source has these relevant lines:

* lines 652--655: `createFetchPatch` returns `originalFetch` only when
  `isAdditionalFetchSupportEnabled()` is false;
* lines 661--678: with additional support enabled, the patch builds a
  dispatcher and, when certificate support is enabled, passes
  `connect: { ca: requestCA }`;
* lines 672--674: `requestCA` is `tls.rootCertificates` plus
  `getOrLoadAdditionalCertificates(params)`;
* lines 1151--1158: additional certificates are supplied by the extension host;
* lines 1191--1201 and 1233--1237: with
  `http.systemCertificatesNode` true, the host loads
  `tls.getCACertificates('system')`;
* lines 1329--1336: the other path reads fixed Linux system CA bundle files;
  neither path reads an arbitrary per-run PEM.

The extension-host configuration mapping at byte offset 2,107,318 is also
explicit: `addCertificatesV1` is enabled by
`http.systemCertificates` when V2 is false, and `addCertificatesV2` is enabled
by the experimental V2 setting when V2 is true. The schema at offsets
609,279--611,251 declares `http.fetchAdditionalSupport` default `true`,
`http.systemCertificates` default `true`, `http.systemCertificatesNode`
default `true`, `http.electronFetch` default `false`, and
`http.proxyStrictSSL` default `true`.

This explains the difference between the two callers. A standalone Node fetch
uses Node's default CA set, which includes the certificate loaded by
`NODE_EXTRA_CA_CERTS`. VS Code's patched fetch uses an explicit Undici
dispatcher whose CA list is built from the bundled and **system** sets. The
per-run certificate is absent from that system set.

As a bounded read-only confirmation on the notebook, with the allowed
certificate path in `NODE_EXTRA_CA_CERTS`, Node v26.8.1 reported:

```text
default 122 true
system 726 false
bundled 121 false
root 121 false
```

The `true`/`false` value compares the SHA256-identified certificate's X.509
fingerprint (`5D:30:F7:ED:8D:9E:BA:39:D9:B0:84:FE:2A:26:0E:7A:11:F5:70:76:95:28:9D:DB:37:93:2E:AE:21:F3:BB:BB`), not PEM whitespace.

## Narrow supported retry

For a fresh disposable VS Code user-data directory, merge these settings into
`User/settings.json` before starting the editor:

```json
{
  "http.fetchAdditionalSupport": false,
  "http.electronFetch": false,
  "http.proxyStrictSSL": true
}
```

`http.fetchAdditionalSupport: false` is the direct source-backed switch: the
proxy-agent wrapper returns the original Node fetch before it adds a dispatcher.
`http.electronFetch: false` makes the intended path explicit. The loopback
manifest is direct, so the tradeoff of disabling VS Code's extra proxy/fetch
handling is bounded to this disposable run. TLS verification remains enabled;
this does not use `NODE_TLS_REJECT_UNAUTHORIZED=0` or
`http.proxyStrictSSL=false`.

If root wants to retain the extra fetch wrapper, a secondary source-backed
probe is `http.systemCertificates: false` (with V2 left false): the mapping then
sets both certificate arms false, and the loopback direct path falls through to
the original fetch. The first setting is preferable because its effect is
unconditional and visible at lines 652--655. Neither setting was live-tested by
this worker; root owns the retry and must record whether the inherited
`NODE_EXTRA_CA_CERTS` reaches the extension host.

There is no installed VS Code/proxy-agent setting that names an arbitrary PEM
file for this patched fetch path. Installing the certificate into the notebook
OS trust store would make the system set contain it, but that is a broader
machine-level change and was not performed. If the isolated setting retry still
fails, the next authorized code change is an explicit CA-aware manifest request
in `runtime.ts`; that is outside this read-only task.

## Remaining gates

The diagnosis does not admit the managed editor route. Root still owns the
fresh retry, managed asset/model identity checks, native request acceptance,
and cleanup. The existing run is terminal with no worker-owned survivors.
