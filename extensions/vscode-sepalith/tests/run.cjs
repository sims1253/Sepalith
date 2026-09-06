const { runTests } = require('@vscode/test-electron');
const path = require('node:path');
const fs = require('node:fs/promises');
const os = require('node:os');
(async () => {
  const root = await fs.mkdtemp(path.join(os.tmpdir(), 'sepalith-editor-'));
  try {
    await fs.mkdir(path.join(root, 'user', 'User'), { recursive: true });
    await fs.writeFile(path.join(root, 'user', 'User', 'settings.json'), JSON.stringify({
      'sepalith.autoStart': false, 'sepalith.debounceMs': 0,
      'editor.inlineSuggest.enabled': true, 'security.workspace.trust.enabled': false,
      'telemetry.telemetryLevel': 'off', 'update.mode': 'none'
    }));
    await runTests({
      version: process.env.SEPALITH_VSCODE_VERSION || '1.104.3',
      extensionDevelopmentPath: path.resolve(__dirname, '..'),
      extensionTestsPath: path.join(__dirname, 'editor.cjs'),
      launchArgs: ['--disable-gpu', '--no-sandbox', '--skip-welcome', '--skip-release-notes',
        '--disable-extensions', '--user-data-dir', path.join(root, 'user'), '--extensions-dir', path.join(root, 'extensions')]
    });
  } finally { await fs.rm(root, { recursive: true, force: true }); }
})().catch(error => { console.error(error); process.exitCode = 1; });
