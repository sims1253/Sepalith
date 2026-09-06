import { createHash, randomUUID } from 'node:crypto';
import * as fs from 'node:fs/promises';
import * as path from 'node:path';
import * as os from 'node:os';
import { createReadStream, createWriteStream } from 'node:fs';
import { Readable, Transform } from 'node:stream';
import type { ReadableStream } from 'node:stream/web';
import { pipeline } from 'node:stream/promises';
import { execFile } from 'node:child_process';
import { promisify } from 'node:util';
import { isIP } from 'node:net';

export function sharedCacheRoot(): string {
  return process.env.SEPALITH_HOME || (process.platform === 'win32'
    ? path.join(process.env.LOCALAPPDATA || path.join(os.homedir(), 'AppData', 'Local'), 'Sepalith')
    : path.join(os.homedir(), '.local', 'share', 'sepalith'));
}

type Backend = 'cpu' | 'vulkan' | 'metal';
interface Asset { name: string; url: string; sha256: string; bytes: number; executable?: boolean }
interface Bundle { platform: string; arch: string; backend: Backend; server: string; files: Asset[] }
export interface Manifest {
  schema: 1; build: string; model: Asset; bundles: Bundle[];
  modelProfile: { renderer: 'zeta2-v1'; task: 'r-next-edit'; tokenizerRevision: string; modelRevision: string };
}

function keysOnly(value: unknown, keys: string[]): boolean {
  return !!value && typeof value === 'object' && !Array.isArray(value) && Object.keys(value).every(k => keys.includes(k));
}
function safeName(value: unknown): boolean {
  return typeof value === 'string' && /^[a-zA-Z0-9][a-zA-Z0-9._-]*$/.test(value) &&
    !value.endsWith('.') && !value.includes('..') && !/^(con|prn|aux|nul|com[1-9]|lpt[1-9])(?:\.|$)/i.test(value);
}
function httpsUrl(value: unknown): boolean {
  if (typeof value !== 'string' || !value.startsWith('https://') || /[^\x21-\x7e]|\\/.test(value)) return false;
  try {
    const u = new URL(value);
    const authority = value.slice(8).split(/[/?#]/)[0];
    if (!authority || authority.includes('@')) return false;
    const match = /^(\[[^\]]+\]|[^:]+)(?::([0-9]+))?$/.exec(authority);
    if (!match || (match[2] !== undefined && (+match[2] < 1 || +match[2] > 65535))) return false;
    const host = match[1];
    if (host.startsWith('[')) return isIP(host.slice(1, -1)) === 6;
    if (isIP(u.hostname) === 4 && host !== u.hostname) return false;
    return host.length <= 253 && host.split('.').every(label => /^[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?$/.test(label));
  }
  catch { return false; }
}
function validAsset(a: Asset): void {
  if (!keysOnly(a, ['name', 'url', 'sha256', 'bytes', 'executable']) || !safeName(a.name) ||
      typeof a.sha256 !== 'string' || !/^[a-f0-9]{64}$/.test(a.sha256) || !Number.isSafeInteger(a.bytes) || a.bytes <= 0 ||
      (a.executable !== undefined && typeof a.executable !== 'boolean') || !httpsUrl(a.url)) throw new Error('Invalid release asset');
}
export function validateManifest(value: unknown): Manifest {
  // SAFETY: this provisional view never escapes until all manifest fields pass the checks below.
  // keysOnly rejects null and primitives before any property access.
  const m = value as Manifest;
  if (!keysOnly(m, ['schema', 'build', 'model', 'bundles', 'modelProfile']) || m.schema !== 1 || !safeName(m.build) || !Array.isArray(m.bundles) || !m.bundles.length)
    throw new Error('Invalid runtime manifest');
  validAsset(m.model);
  const profile = m.modelProfile;
  if (!keysOnly(profile, ['renderer', 'task', 'tokenizerRevision', 'modelRevision']) || profile.renderer !== 'zeta2-v1' || profile.task !== 'r-next-edit' ||
      typeof profile.tokenizerRevision !== 'string' || !profile.tokenizerRevision.trim() ||
      typeof profile.modelRevision !== 'string' || !profile.modelRevision.trim())
    throw new Error('Unsupported model profile: require zeta2-v1/r-next-edit and pinned model/tokenizer revisions');
  const keys = new Set<string>();
  for (const b of m.bundles) {
    if (!keysOnly(b, ['platform', 'arch', 'backend', 'server', 'files'])) throw new Error('Invalid runtime bundle');
    const key = `${b.platform}-${b.arch}-${b.backend}`;
    if (!['linux', 'win32', 'darwin'].includes(b.platform) || !['x64', 'arm64'].includes(b.arch) ||
        !['cpu', 'vulkan', 'metal'].includes(b.backend) || !Array.isArray(b.files) || keys.has(key))
      throw new Error('Invalid or duplicate runtime bundle');
    keys.add(key);
    b.files.forEach(validAsset);
    if (new Set(b.files.map(a => a.name.toLowerCase())).size !== b.files.length || !b.files.some(a => a.name === b.server && a.executable === true))
      throw new Error('Bundle must contain a unique executable server');
  }
  return m;
}

export async function detectBackend(): Promise<Backend> {
  if (process.platform === 'darwin') return 'metal';
  // A working Vulkan device is stronger evidence than a GPU vendor name.
  try {
    const { stdout } = await promisify(execFile)('vulkaninfo', ['--summary'], { timeout: 5000, maxBuffer: 1024 * 1024 });
    if (/deviceType\s*=\s*PHYSICAL_DEVICE_TYPE_(DISCRETE|INTEGRATED)_GPU/.test(stdout)) return 'vulkan';
  } catch { /* Try OS hardware inventory when Vulkan tools are absent. */ }
  try {
    if (process.platform === 'linux') {
      for (const entry of await fs.readdir('/sys/class/drm')) {
        if (!/^card[0-9]+$/.test(entry)) continue;
        const vendor = (await fs.readFile(`/sys/class/drm/${entry}/device/vendor`, 'utf8')).trim();
        if (['0x10de', '0x1002', '0x8086'].includes(vendor)) return 'vulkan';
      }
    } else if (process.platform === 'win32') {
      const { stdout } = await promisify(execFile)('powershell.exe', ['-NoProfile', '-NonInteractive', '-Command',
        'Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty PNPDeviceID'], { timeout: 5000 });
      if (/VEN_(10DE|1002|8086)/i.test(stdout)) return 'vulkan';
    }
  } catch { /* Missing inventory is a working CPU configuration. */ }
  return 'cpu';
}
export function selectBundle(m: Manifest, platform: string, arch: string, backend: Backend): Bundle {
  const candidates = m.bundles.filter(b => b.platform === platform && b.arch === arch);
  const selected = candidates.find(b => b.backend === backend) ?? candidates.find(b => b.backend === 'cpu')
    ?? (platform === 'darwin' && backend === 'cpu' ? candidates.find(b => b.backend === 'metal') : undefined);
  if (!selected) throw new Error(`No runtime for ${platform}/${arch}/${backend}`);
  return selected;
}
async function checksum(file: string, signal: AbortSignal): Promise<string> {
  const hash = createHash('sha256');
  for await (const chunk of createReadStream(file)) { signal.throwIfAborted(); hash.update(chunk); }
  return hash.digest('hex');
}
async function withDeadline<T>(signal: AbortSignal, milliseconds: number,
                               action: (bounded: AbortSignal) => Promise<T>): Promise<T> {
  signal.throwIfAborted();
  if (!Number.isFinite(milliseconds) || milliseconds <= 0) throw new Error('Invalid network deadline');
  const controller = new AbortController();
  const abort = () => controller.abort(signal.reason);
  signal.addEventListener('abort', abort, { once: true });
  const timer = setTimeout(() => controller.abort(new Error('Network deadline exceeded')), milliseconds);
  try { return await action(controller.signal); }
  finally { clearTimeout(timer); signal.removeEventListener('abort', abort); }
}

export async function install(a: Asset, directory: string, signal: AbortSignal, timeoutMs = 600_000): Promise<string> {
  validAsset(a);
  signal.throwIfAborted();
  await fs.mkdir(directory, { recursive: true });
  const target = path.join(directory, a.name);
  try {
    if ((await fs.stat(target)).size === a.bytes && await checksum(target, signal) === a.sha256) {
      signal.throwIfAborted();
      if (a.executable && process.platform !== 'win32') await fs.chmod(target, 0o755);
      signal.throwIfAborted();
      return target;
    }
  } catch { signal.throwIfAborted(); /* Cache miss. */ }
  const temporary = `${target}.${randomUUID()}.partial`;
  try { return await withDeadline(signal, timeoutMs, async bounded => {
    const response = await fetch(a.url, { signal: bounded });
    if (!response.ok || !response.body || new URL(response.url).protocol !== 'https:') throw new Error(`Download failed: ${a.name} (${response.status})`);
    const hash = createHash('sha256');
    let bytes = 0;
    const verify = new Transform({ transform(chunk, _encoding, callback) {
      bytes += chunk.length;
      if (bytes > a.bytes) return callback(new Error(`Asset exceeds expected size: ${a.name}`));
      hash.update(chunk); callback(null, chunk);
    }});
    // SAFETY: fetch supplies a WHATWG byte stream; Node 18's DOM and node:stream/web declarations differ.
    await pipeline(Readable.fromWeb(response.body as ReadableStream<Uint8Array>), verify, createWriteStream(temporary, { flags: 'wx' }), { signal: bounded });
    if (bytes !== a.bytes || hash.digest('hex') !== a.sha256) throw new Error(`Checksum or size mismatch: ${a.name}`);
    if (a.executable && process.platform !== 'win32') await fs.chmod(temporary, 0o755);
    bounded.throwIfAborted();
    await fs.rename(temporary, target);
    return target;
  }); } finally { await fs.rm(temporary, { force: true }); }
}

export interface ProvisionOptions { refreshManifest?: boolean; manifestTimeoutMs?: number; assetTimeoutMs?: number }

// Cached manifests are validated again on every use. Refresh is explicit; a failed
// refresh leaves the previous cache untouched and reports the failure to its caller.
export async function loadManifest(url: string, storage: string, signal: AbortSignal,
                                   options: ProvisionOptions = {}): Promise<Manifest> {
  signal.throwIfAborted();
  if (!httpsUrl(url)) throw new Error('Release manifest must use HTTPS without credentials');
  const directory = path.join(storage, 'manifests');
  const file = path.join(directory, createHash('sha256').update(url).digest('hex') + '.json');
  if (!options.refreshManifest) {
    try {
      const cached = validateManifest(JSON.parse(await fs.readFile(file, 'utf8')));
      signal.throwIfAborted();
      return cached;
    } catch (error) {
      signal.throwIfAborted();
      if (!(error instanceof Error && 'code' in error && error.code === 'ENOENT'))
        throw new Error('Cached manifest is invalid; explicitly refresh the release manifest');
    }
  }
  const manifest = await withDeadline(signal, options.manifestTimeoutMs ?? 15_000, async bounded => {
    const response = await fetch(url, { signal: bounded });
    if (!response.ok || new URL(response.url).protocol !== 'https:') throw new Error(`Manifest request failed: ${response.status}`);
    const value = validateManifest(await response.json());
    bounded.throwIfAborted();
    return value;
  });
  await fs.mkdir(directory, { recursive: true });
  const temporary = `${file}.${randomUUID()}.partial`;
  try {
    await fs.writeFile(temporary, JSON.stringify(manifest), { flag: 'wx' });
    signal.throwIfAborted();
    await fs.rename(temporary, file);
  } finally { await fs.rm(temporary, { force: true }); }
  return manifest;
}

export async function provision(manifestUrl: string, storage: string, backend: Backend | 'auto', signal: AbortSignal,
                                log: (message: string) => void, modelOverride = '', options: ProvisionOptions = {}) {
  const m = await loadManifest(manifestUrl, storage, signal, options);
  if (modelOverride) {
    if ((await fs.stat(modelOverride)).size !== m.model.bytes || await checksum(modelOverride, signal) !== m.model.sha256)
      throw new Error('Model override differs from the release manifest; use an explicit manual serverPath for development');
    signal.throwIfAborted();
  }
  const selected = selectBundle(m, process.platform, process.arch, backend === 'auto' ? await detectBackend() : backend);
  // Hashes partition caches when a release is replaced or a model changes.
  const identity = createHash('sha256').update(JSON.stringify(selected)).digest('hex').slice(0, 16);
  const runtime = path.join(storage, 'runtimes', m.build, identity);
  for (const asset of selected.files) {
    log(`Checking runtime asset ${asset.name}`);
    await install(asset, runtime, signal, options.assetTimeoutMs);
  }
  const model = modelOverride || await install(m.model, path.join(storage, 'models', m.model.sha256), signal, options.assetTimeoutMs);
  signal.throwIfAborted();
  return { serverPath: path.join(runtime, selected.server), modelPath: model, gpuLayers: backend === 'cpu' || selected.backend === 'cpu' ? 0 : 99 };
}
