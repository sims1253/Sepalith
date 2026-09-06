const assert = require('node:assert/strict');
const http = require('node:http');
const vscode = require('vscode');
const { SepalithProvider } = require('../dist/extension.js');
const delay = ms => new Promise(resolve => setTimeout(resolve, ms));
async function until(check) {
  const end = Date.now() + 10000;
  while (!check()) { if (Date.now() > end) throw Error('Timed out'); await delay(25); }
}
exports.run = async function () {
  let mode = 'ok', count = 0, pending = [];
  const server = http.createServer((req, res) => {
    if (req.url === '/health') { res.end('{}'); return; }
    let body = '';
    req.on('data', chunk => body += chunk);
    req.on('end', () => {
      const prompt = JSON.parse(body).prompt;
      const respond = () => {
        res.setHeader('Content-Type', 'application/json');
        res.end(JSON.stringify({ choices: [{ text: 'answer <- sum(values)' }], usage: { completion_tokens: 5 } }));
      };
      if (prompt === 'x') { respond(); return; }
      count++;
      if (mode === 'hold') pending.push(respond);
      else if (mode === 'error') { res.writeHead(503); res.end('unavailable'); }
      else respond();
    });
  });
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  try {
    const config = vscode.workspace.getConfiguration('sepalith');
    for (const [key, value] of Object.entries({ autoStart: false, debounceMs: 0, scopeContext: false, port: server.address().port }))
      await config.update(key, value, vscode.ConfigurationTarget.Global);
    await vscode.extensions.getExtension('sepalith-dev.vscode-sepalith').activate();
    await vscode.commands.executeCommand('sepalith.startServer');
    const doc = await vscode.workspace.openTextDocument({ language: 'r', content: '# A deterministic editor integration fixture\nvalues <- c(1, 2, 3)\nanswer <- ' });
    const editor = await vscode.window.showTextDocument(doc);
    const position = doc.lineAt(2).range.end;
    editor.selection = new vscode.Selection(position, position);
    const provider = new SepalithProvider();
    const context = { triggerKind: vscode.InlineCompletionTriggerKind.Invoke };
    const request = (token = new vscode.CancellationTokenSource().token) => provider.provideInlineCompletionItems(doc, position, context, token);

    const items = await request();
    assert.equal(items.length, 1);
    assert.equal(items[0].range.start.character, 0);
    const before = count;
    assert.deepEqual(await request(), items);
    assert.equal(count, before, 'unchanged request should use cache');
    console.log('PASS provider replacement and cache');

    provider.invalidate(); mode = 'hold';
    const cancel = new vscode.CancellationTokenSource();
    const cancelled = request(cancel.token);
    await until(() => pending.length === 1); cancel.cancel();
    assert.deepEqual(await cancelled, []); pending.shift()(); cancel.dispose();
    mode = 'ok'; assert.equal((await request()).length, 1);
    console.log('PASS cancellation and same-prompt retry');

    provider.invalidate(); mode = 'hold';
    const stale = request(); await until(() => pending.length === 1);
    await editor.edit(edit => edit.insert(doc.lineAt(0).range.end, ' changed'));
    pending.shift()(); assert.deepEqual(await stale, []);
    console.log('PASS response rejected after document edit');

    provider.invalidate(); mode = 'error'; assert.deepEqual(await request(), []);
    mode = 'ok'; await delay(100); assert.equal((await request()).length, 1);
    console.log('PASS server error and retry');

    // Exercise the registered provider and VS Code's actual accept/undo commands.
    editor.selection = new vscode.Selection(position, position);
    const original = doc.getText(); const initialCount = count;
    await vscode.commands.executeCommand('editor.action.inlineSuggest.trigger');
    await until(() => count > initialCount); await delay(500);
    const accepted = SepalithProvider.accepted;
    await config.update('debounceMs', 200, vscode.ConfigurationTarget.Global);
    await vscode.commands.executeCommand('editor.action.inlineSuggest.commit');
    await until(() => doc.getText() !== original);
    await delay(400);
    assert.equal(SepalithProvider.accepted, accepted + 1);
    assert.equal(count, initialCount + 1, 'accept must not schedule another suggestion');
    await config.update('debounceMs', 0, vscode.ConfigurationTarget.Global);
    const beforeManual = count;
    await vscode.commands.executeCommand('sepalith.suggest');
    await until(() => count > beforeManual);
    await vscode.commands.executeCommand('editor.action.inlineSuggest.hide');
    assert.equal(doc.lineAt(2).text, 'answer <- sum(values)');
    await vscode.commands.executeCommand('undo');
    await until(() => doc.getText() === original);
    console.log('PASS actual inline accept and undo');

    mode = 'hold';
    void vscode.commands.executeCommand('editor.action.inlineSuggest.trigger');
    // The prior result can be cached; change the document to require a new request.
    await editor.edit(edit => edit.insert(doc.lineAt(0).range.end, ' switch'));
    void vscode.commands.executeCommand('editor.action.inlineSuggest.trigger');
    await until(() => pending.length > 0);
    const other = await vscode.workspace.openTextDocument({ language: 'r', content: '# Another file must not receive an old suggestion\nother <- 1\n' });
    await vscode.window.showTextDocument(other);
    for (const respond of pending.splice(0)) respond();
    await delay(250);
    await vscode.commands.executeCommand('editor.action.inlineSuggest.commit');
    assert.equal(other.getText(), '# Another file must not receive an old suggestion\nother <- 1\n');
    console.log('PASS file switch discards pending suggestion');
    await vscode.commands.executeCommand('sepalith.stopServer');
    assert.equal((await fetch(`http://127.0.0.1:${server.address().port}/health`)).status, 200);
    console.log('PASS stop preserves external server');
    mode = 'ok';
    await vscode.commands.executeCommand('sepalith.startServer');
    const restarted = new SepalithProvider();
    const result = await restarted.provideInlineCompletionItems(other, other.lineAt(1).range.end, context, new vscode.CancellationTokenSource().token);
    assert.equal(result.length, 1);
    console.log('PASS extension reconnects after stop');
  } finally {
    for (const respond of pending) respond();
    server.closeAllConnections(); await new Promise(resolve => server.close(resolve));

  }
};
