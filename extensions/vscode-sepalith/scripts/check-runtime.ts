import * as assert from 'node:assert/strict';
import { selectBundle, validateManifest, loadManifest, provision } from '../src/runtime';
const fixtures: {cases: {name: string; valid: boolean; manifest: unknown}[]} = require('../../../tests/fixtures/release-manifests.json');
for (const example of fixtures.cases) {
  if (example.valid) assert.doesNotThrow(() => validateManifest(example.manifest), example.name);
  else assert.throws(() => validateManifest(example.manifest), example.name);
}
console.log(`${fixtures.cases.length} shared release manifest fixtures passed`);
const asset = { name: 'llama-server', url: 'https://example.com/server', sha256: 'a'.repeat(64), bytes: 100, executable: true };
const manifest = { schema: 1, build: 'b10453', modelProfile: {renderer: 'zeta2-v1', task: 'r-next-edit', tokenizerRevision: 'tokenizer@sha256:fixture', modelRevision: 'model@sha256:fixture'}, model: {...asset, name: 'model.gguf'}, bundles: [
  { platform: 'linux', arch: 'x64', backend: 'cpu', server: asset.name, files: [asset] },
  { platform: 'darwin', arch: 'arm64', backend: 'metal', server: asset.name, files: [asset] },
] };
const valid = validateManifest(manifest);
assert.equal(selectBundle(valid, 'linux', 'x64', 'vulkan').backend, 'cpu');
assert.equal(selectBundle(valid, 'darwin', 'arm64', 'metal').backend, 'metal');
assert.equal(selectBundle(valid, 'darwin', 'arm64', 'cpu').backend, 'metal');
assert.throws(() => selectBundle(valid, 'win32', 'arm64', 'cpu'));
for (const name of ['../evil', '/evil', 'C:\\evil', '.', '..']) {
  assert.throws(() => validateManifest({...manifest, model: {...asset, name}}));
}
assert.throws(() => validateManifest({...manifest, model: {...asset, sha256: 'bad'}}));
assert.throws(() => validateManifest({...manifest, model: {...asset, url: 'http://example.com'}}));
assert.throws(() => validateManifest({...manifest, bundles: [...manifest.bundles, manifest.bundles[0]]}));
assert.throws(() => validateManifest({...manifest, bundles: [{...manifest.bundles[0], server: 'missing'}]}));
console.log('Runtime manifest and platform checks passed');

import { install } from '../src/runtime';
import { createHash } from 'node:crypto';
import * as fs from 'node:fs/promises';
import * as os from 'node:os';
import * as path from 'node:path';
async function testDownloads() {
  const dir = await fs.mkdtemp(path.join(os.tmpdir(), 'sepalith-download-'));
  const originalFetch = globalThis.fetch;
  let calls = 0;
  let payload = Buffer.from('verified bytes');
  const a = {...asset, bytes: payload.length, sha256: createHash('sha256').update(payload).digest('hex')};
  globalThis.fetch = async () => {
    calls++;
    const response = new Response(payload);
    Object.defineProperty(response, 'url', {value: a.url});
    return response;
  };
  try {
    const target = await install(a, dir, new AbortController().signal);
    await install(a, dir, new AbortController().signal);
    assert.equal(calls, 1, 'verified cache must avoid network');
    const cachedCancel = new AbortController(); cachedCancel.abort();
    await assert.rejects(install(a, dir, cachedCancel.signal), 'canceled cache hit must not succeed');
    await fs.writeFile(target, 'tampered');
    await install(a, dir, new AbortController().signal);
    assert.equal(calls, 2, 'tampered cache must be replaced');
    await fs.rm(target);
    payload = Buffer.from('bad');
    await assert.rejects(install(a, dir, new AbortController().signal), /mismatch/);
    assert.deepEqual(await fs.readdir(dir), [], 'failed downloads leave no partial file');
    const controller = new AbortController(); controller.abort();
    await assert.rejects(install(a, dir, controller.signal));
  } finally { globalThis.fetch = originalFetch; await fs.rm(dir, {recursive: true, force: true}); }
  console.log('Download integrity, cache repair, and cancellation checks passed');
}
async function testManifestCache() {
  const dir = await fs.mkdtemp(path.join(os.tmpdir(), 'sepalith-manifest-'));
  const originalFetch = globalThis.fetch;
  const signal = new AbortController().signal;
  const url = 'https://example.com/release.json';
  let calls = 0;
  let value = manifest;
  const response = (body: unknown) => {
    const r = new Response(JSON.stringify(body));
    Object.defineProperty(r, 'url', {value: url}); return r;
  };
  globalThis.fetch = async () => { calls++; return response(value); };
  try {
    await loadManifest(url, dir, signal);
    globalThis.fetch = async () => { throw new Error('offline'); };
    assert.deepEqual(await loadManifest(url, dir, signal), manifest, 'offline restart uses validated cache');
    await assert.rejects(loadManifest(url, dir, signal, {refreshManifest: true}), /offline/);
    assert.deepEqual(await loadManifest(url, dir, signal), manifest, 'failed refresh preserves previous manifest');
    globalThis.fetch = async () => { calls++; return response(value); };
    value = {...manifest, modelProfile: {...manifest.modelProfile, renderer: 'psm'}};
    await assert.rejects(loadManifest(url, dir, signal, {refreshManifest: true}), /profile/);
    assert.deepEqual(await loadManifest(url, dir, signal), manifest);
    value = {...manifest, build: 'b10454'};
    await loadManifest(url, dir, signal, {refreshManifest: true});
    assert.equal((await loadManifest(url, dir, signal)).build, 'b10454');
    const canceled = new AbortController(); canceled.abort();
    await assert.rejects(loadManifest(url, dir, canceled.signal));
    const files = await fs.readdir(path.join(dir, 'manifests'));
    assert.equal(files.length, 1);
    await fs.writeFile(path.join(dir, 'manifests', files[0]), '{}');
    await assert.rejects(loadManifest(url, dir, signal), /Cached manifest is invalid/);
    await loadManifest(url, dir, signal, {refreshManifest: true});
    const override = path.join(dir, 'override.gguf'); await fs.writeFile(override, 'wrong model');
    await assert.rejects(provision(url, dir, 'cpu', signal, () => {}, override), /manual serverPath/);
    assert.equal(calls, 4, 'cache hits do not fetch');
    globalThis.fetch = (_url, options) => new Promise((_resolve, reject) => {
      const bounded = options?.signal;
      bounded?.addEventListener('abort', () => reject(bounded.reason), {once: true});
    });
    await assert.rejects(loadManifest(url, dir, signal, {refreshManifest: true, manifestTimeoutMs: 10}), /deadline/);
    await assert.rejects(install(asset, path.join(dir, 'timeout'), signal, 10), /deadline/);
    assert.deepEqual(await fs.readdir(path.join(dir, 'timeout')), []);
  } finally { globalThis.fetch = originalFetch; await fs.rm(dir, {recursive: true, force: true}); }
  console.log('Manifest offline restart, refresh, rejection, cancellation, override and deadlines passed');
}
async function testOfflineProvision() {
  const dir = await fs.mkdtemp(path.join(os.tmpdir(), 'sepalith-provision-'));
  const originalFetch = globalThis.fetch;
  const payload = Buffer.from('inert fixture bytes, never executed');
  const a = {...asset, bytes: payload.length, sha256: createHash('sha256').update(payload).digest('hex')};
  const release = {...manifest, model: {...a, name: 'model.gguf'}, bundles: [
    {platform: process.platform, arch: process.arch, backend: 'cpu', server: a.name, files: [a]},
  ]};
  const signal = new AbortController().signal;
  const url = 'https://example.com/offline-release.json';
  let calls = 0;
  globalThis.fetch = async input => {
    calls++;
    const r = new Response(String(input) === url ? JSON.stringify(release) : payload);
    Object.defineProperty(r, 'url', {value: String(input)}); return r;
  };
  try {
    const installed = await provision(url, dir, 'cpu', signal, () => {});
    assert.equal(calls, 3);
    globalThis.fetch = async () => { throw new Error('offline'); };
    assert.deepEqual(await provision(url, dir, 'cpu', signal, () => {}), installed);
    assert.equal((await provision(url, dir, 'cpu', signal, () => {}, installed.modelPath)).modelPath, installed.modelPath);
    assert.equal(installed.gpuLayers, 0);
  } finally { globalThis.fetch = originalFetch; await fs.rm(dir, {recursive: true, force: true}); }
  console.log('Complete offline provisioning and verified model override passed (no executable launched)');
}
async function main() { await testDownloads(); await testManifestCache(); await testOfflineProvision(); }
main().catch(error => { console.error(error); process.exitCode = 1; });
