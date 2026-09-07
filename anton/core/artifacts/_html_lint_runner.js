// Checked-in runner for html_lint.py (ENG-1204 Fix 3) — not agent-authored.
// Target file path comes via ANTON_HTML_LINT_TARGET (env, not argv, to
// avoid mixing Chromium switches with a positional file argument).
//
// Loads the page in a hidden, sandboxed BrowserWindow and reports one JSON
// blob between RESULT_JSON_START/RESULT_JSON_END on stdout: console errors
// scoped to the target's own directory (excludes Electron's own internal
// diagnostics, e.g. its CSP warning — filtered by source, not content),
// failed local (file://) asset requests, and whether the renderer crashed.
//
// Network policy (onBeforeRequest below): runs automatically per cell,
// unlike a human opening the artifact, so it's tighter than normal runtime.
// Not fully closed — an external GET can still leak a little via query
// params; same order of risk the product already accepts on preview.

'use strict';

const { app, BrowserWindow } = require('electron');
const path = require('path');
const { pathToFileURL } = require('url');

app.disableHardwareAcceleration();

const SETTLE_MS = 500;
const HARD_TIMEOUT_MS = 15000; // safety net; the real enforcement is the caller's subprocess timeout

function emitAndExit(result) {
  process.stdout.write('RESULT_JSON_START\n');
  process.stdout.write(JSON.stringify(result) + '\n');
  process.stdout.write('RESULT_JSON_END\n');
  app.exit(0);
}

app.whenReady().then(() => {
  const targetPath = process.env.ANTON_HTML_LINT_TARGET;
  const result = { consoleErrors: [], failedRequests: [], crashed: false, crashDetails: null };

  if (!targetPath) {
    emitAndExit(result);
    return;
  }

  let targetDirUrl = pathToFileURL(path.dirname(path.resolve(targetPath))).href;
  if (!targetDirUrl.endsWith('/')) targetDirUrl += '/';

  const win = new BrowserWindow({
    show: false,
    webPreferences: {
      offscreen: true,
      sandbox: true,
      contextIsolation: true,
      nodeIntegration: false,
      webSecurity: true,
    },
  });

  const hardTimeout = setTimeout(() => emitAndExit(result), HARD_TIMEOUT_MS);

  win.webContents.on('console-message', (_event, level, message, line, sourceId) => {
    // level 3 == error in Electron's console-message event.
    if (level !== 3) return;
    if (!sourceId || !sourceId.startsWith(targetDirUrl)) return;
    result.consoleErrors.push({ message, line });
  });

  win.webContents.on('render-process-gone', (_event, details) => {
    result.crashed = true;
    result.crashDetails = details;
    clearTimeout(hardTimeout);
    emitAndExit(result);
  });

  win.webContents.session.webRequest.onErrorOccurred((details) => {
    if (details.url.startsWith(targetDirUrl)) {
      result.failedRequests.push({ url: details.url, error: details.error });
    }
  });

  win.webContents.session.webRequest.onBeforeRequest((details, callback) => {
    let url;
    try {
      url = new URL(details.url);
    } catch {
      callback({});
      return;
    }
    const isLocal = url.protocol === 'file:';
    const isLoopback = url.hostname === 'localhost' || url.hostname === '127.0.0.1' || url.hostname === '::1';
    if (isLocal || isLoopback || details.method === 'GET') {
      callback({});
      return;
    }
    // External, state-changing request: neutralize as an empty SUCCESS
    // rather than a cancel, so a correctly-written artifact's own
    // `.catch(...)` doesn't fire and log a spurious console error that
    // would be an artifact of our interception, not a real bug.
    callback({ redirectURL: 'data:text/plain,' });
  });

  win
    .loadFile(targetPath)
    .catch(() => {})
    .finally(() => {
      setTimeout(() => {
        clearTimeout(hardTimeout);
        emitAndExit(result);
      }, SETTLE_MS);
    });
});

app.on('window-all-closed', () => {});
