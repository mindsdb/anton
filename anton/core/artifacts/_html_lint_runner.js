// Checked-in runner for html_lint.py — not agent-authored.
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

// Heuristic, not authoritative: true only when the body has neither visible
// text nor a non-script/style/meta child element. Catches "nothing rendered
// at all" (e.g. a page whose entire markup got swallowed as text by an
// unclosed <title>/<script>/<style> — real case found in testing) without
// flaging a legitimately content-free page that fills in later (async data,
// a splash screen). False negatives are expected (e.g. a blank <canvas>).
const EMPTY_PAGE_CHECK = `(() => {
  if (!document.body) return true;
  const NON_VISUAL = new Set(['SCRIPT', 'STYLE', 'TEMPLATE', 'LINK', 'META', 'NOSCRIPT', 'TITLE']);
  const visibleChildren = Array.from(document.body.children).filter((el) => !NON_VISUAL.has(el.tagName));
  const hasText = document.body.innerText.trim().length > 0;
  return visibleChildren.length === 0 && !hasText;
})()`;

app.whenReady().then(() => {
  const targetPath = process.env.ANTON_HTML_LINT_TARGET;
  const result = { consoleErrors: [], failedRequests: [], crashed: false, crashDetails: null, emptyPage: false };

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

  win.webContents.on('console-message', (event) => {
    // Single-event-object form (the older (event, level, message, line,
    // sourceId) callback is deprecated and warns on stderr as of this
    // Electron). `event.level` is the string "error"/"warning"/"info" here,
    // not the old numeric level.
    if (event.level !== 'error') return;
    if (!event.sourceId || !event.sourceId.startsWith(targetDirUrl)) return;
    result.consoleErrors.push({ message: event.message, line: event.lineNumber });
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
      setTimeout(async () => {
        clearTimeout(hardTimeout);
        if (!result.crashed) {
          result.emptyPage = await win.webContents.executeJavaScript(EMPTY_PAGE_CHECK).catch(() => false);
        }
        emitAndExit(result);
      }, SETTLE_MS);
    });
});

app.on('window-all-closed', () => {});
