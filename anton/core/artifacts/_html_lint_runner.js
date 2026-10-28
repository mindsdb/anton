// Checked-in runner for html_lint.py — not agent-authored.
// Target comes via ANTON_HTML_LINT_TARGET (env, not argv, to avoid mixing
// Chromium switches with a positional argument): a file path, loaded through
// file://, or an http(s) URL — the fullstack page served by its own running
// backend, where the relative /api/* fetches resolve.
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
// Probe mode (ANTON_HTML_LINT_PROBE=1) also measures the page at two widths,
// takes a screenshot and presses a few controls: what the artifact builder
// reads to revise its draft. Plain lint mode is unchanged.
const PROBE = process.env.ANTON_HTML_LINT_PROBE === '1';
const DESKTOP = { width: 1280, height: 900 };
const PHONE = { width: 390, height: 844 };
const MAX_PRESSES = 12;
const PRESS_SETTLE_MS = 250;
const HARD_TIMEOUT_MS = PROBE ? 22000 : 15000; // safety net; the real enforcement is the caller's subprocess timeout

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

// What the page shows after its scripts ran, at the current width. Only
// visible elements count; a visual smaller than 40 px is an icon, not a view.
const OUTLINE = `(() => {
  const vw = window.innerWidth;
  const clip = (s, n) => { s = (s || '').replace(/\\s+/g, ' ').trim(); return s.length > n ? s.slice(0, n - 1) + '…' : s; };
  const visible = (el) => { const r = el.getBoundingClientRect(); const cs = getComputedStyle(el); return r.width > 0 && r.height > 0 && cs.visibility !== 'hidden' && cs.display !== 'none'; };
  const name = (el) => el.id ? '#' + el.id : (el.getAttribute('aria-label') || el.tagName.toLowerCase());
  const scrolls = (el) => { for (let p = el.parentElement; p && p !== document.body; p = p.parentElement) { const o = getComputedStyle(p).overflowX; if (o === 'auto' || o === 'scroll' || o === 'hidden') return true; } return false; };
  const all = Array.from(document.body ? document.body.querySelectorAll('*') : []);
  const overflowing = [];
  for (const el of all) {
    if (overflowing.length >= 5) break;
    const r = el.getBoundingClientRect();
    if (r.right > vw + 1 && r.width > 0 && visible(el) && !scrolls(el) && !(el.parentElement && el.parentElement.getBoundingClientRect().right > vw + 1)) {
      overflowing.push(name(el) + ' ' + Math.round(r.width) + 'px wide');
    }
  }
  const headings = Array.from(document.querySelectorAll('h1,h2,h3')).filter(visible).slice(0, 20)
    .map((h) => ({ level: Number(h.tagName[1]), text: clip(h.innerText, 80) }));
  const tables = Array.from(document.querySelectorAll('table')).filter(visible).slice(0, 10).map((t) => {
    const head = t.tHead ? t.tHead.rows[0] : t.rows[0];
    const columns = head ? Array.from(head.cells).slice(0, 8).map((c) => clip(c.innerText, 30)) : [];
    const body = t.tBodies.length ? Array.from(t.tBodies).flatMap((b) => Array.from(b.rows)) : Array.from(t.rows).slice(1);
    const caption = t.caption ? clip(t.caption.innerText, 60) : '';
    return { name: name(t), caption, columns, rows: body.filter(visible).length };
  });
  const visuals = [];
  for (const el of document.querySelectorAll('canvas, svg, img')) {
    if (visuals.length >= 15) break;
    if (el.tagName.toLowerCase() === 'svg' && el.parentElement && el.parentElement.closest('svg')) continue;
    const r = el.getBoundingClientRect();
    if (r.width < 40 || r.height < 40 || !visible(el)) continue;
    let drawn = true;
    const kind = el.tagName.toLowerCase();
    if (kind === 'canvas') {
      try { const blank = document.createElement('canvas'); blank.width = el.width; blank.height = el.height; drawn = el.toDataURL() !== blank.toDataURL(); } catch (e) { drawn = true; }
    } else if (kind === 'svg') {
      drawn = el.querySelectorAll('path,rect,circle,ellipse,line,polyline,polygon,text,image,use').length > 0;
    } else {
      drawn = el.complete && el.naturalWidth > 0;
    }
    const holder = el.closest('[id]');
    visuals.push({ kind, name: el.id ? '#' + el.id : (holder ? 'in #' + holder.id : kind), width: Math.round(r.width), height: Math.round(r.height), drawn });
  }
  let smallText = 0;
  for (const el of all) {
    if (!el.childNodes.length || !Array.from(el.childNodes).some((n) => n.nodeType === 3 && n.textContent.trim())) continue;
    if (visible(el) && parseFloat(getComputedStyle(el).fontSize) < 11) smallText++;
  }
  const failed = Array.from(document.querySelectorAll('[data-view-failed]')).map((el) => el.getAttribute('data-view-failed'));
  return {
    width: vw, scrollWidth: document.documentElement.scrollWidth, overflowing,
    title: document.title, textChars: document.body ? document.body.innerText.trim().length : 0,
    headings, tables, visuals, smallText, failedViews: failed,
  };
})()`;

// Labels of the controls a press pass may try, in page order.
const CONTROLS = `(() => {
  const visible = (el) => { const r = el.getBoundingClientRect(); return r.width > 0 && r.height > 0 && getComputedStyle(el).visibility !== 'hidden'; };
  const sel = 'button, [role=tab], [role=button], select, input[type=checkbox], input[type=radio], input[type=range], summary';
  const found = Array.from(document.querySelectorAll(sel)).filter((el) => visible(el) && !el.disabled);
  window.__antonProbeControls = found;
  return found.map((el) => (el.getAttribute('aria-label') || el.innerText || el.value || el.name || el.id || el.tagName).replace(/\\s+/g, ' ').trim().slice(0, 40));
})()`;

// Press control I and say whether what the page shows changed. Forms are
// not submitted: a submit would navigate the frame away.
const pressControl = (i) => `(async () => {
  const el = (window.__antonProbeControls || [])[${i}];
  if (!el || !el.isConnected) return { pressed: false };
  const sig = () => {
    const text = document.body.innerText;
    let shapes = document.querySelectorAll('svg *').length;
    let pixels = 0;
    for (const c of document.querySelectorAll('canvas')) { try { pixels += c.toDataURL().length; } catch (e) {} }
    const shown = Array.from(document.body.querySelectorAll('*')).filter((e) => e.getClientRects().length).length;
    return text.length + ':' + text.slice(0, 2000) + ':' + shapes + ':' + pixels + ':' + shown;
  };
  const before = sig();
  const block = (e) => e.preventDefault();
  document.addEventListener('submit', block, true);
  try {
    const tag = el.tagName.toLowerCase();
    if (tag === 'select') {
      if (el.options.length > 1) { el.selectedIndex = (el.selectedIndex + 1) % el.options.length; }
      el.dispatchEvent(new Event('input', { bubbles: true }));
      el.dispatchEvent(new Event('change', { bubbles: true }));
    } else if (tag === 'input' && el.type === 'range') {
      const step = Number(el.step) || 1;
      el.value = Number(el.value) + step <= Number(el.max || 100) ? Number(el.value) + step : Number(el.min || 0);
      el.dispatchEvent(new Event('input', { bubbles: true }));
      el.dispatchEvent(new Event('change', { bubbles: true }));
    } else {
      el.click();
    }
    await new Promise((r) => setTimeout(r, ${PRESS_SETTLE_MS}));
    return { pressed: true, changed: sig() !== before };
  } finally {
    document.removeEventListener('submit', block, true);
  }
})()`;

const wait = (ms) => new Promise((r) => setTimeout(r, ms));

async function runProbe(win, result) {
  const js = (code) => win.webContents.executeJavaScript(code, true);
  const probe = { desktop: null, phone: null, controls: [], screenshot: null };
  probe.desktop = await js(OUTLINE).catch(() => null);
  try {
    const image = await win.webContents.capturePage();
    const size = image.getSize();
    const scaled = size.width > 1024 ? image.resize({ width: 1024 }) : image;
    if (!scaled.isEmpty()) probe.screenshot = scaled.toJPEG(70).toString('base64');
  } catch (e) {}
  win.setContentSize(PHONE.width, PHONE.height);
  await wait(300);
  probe.phone = await js(OUTLINE).catch(() => null);
  win.setContentSize(DESKTOP.width, DESKTOP.height);
  await wait(300);
  const labels = await js(CONTROLS).catch(() => []);
  const errorsBefore = result.consoleErrors.length;
  for (let i = 0; i < Math.min(labels.length, MAX_PRESSES); i++) {
    if (result.crashed) break;
    const seen = result.consoleErrors.length;
    const outcome = await js(pressControl(i)).catch((e) => ({ pressed: true, error: String(e && e.message || e) }));
    if (!outcome || !outcome.pressed) continue;
    const errors = result.consoleErrors.slice(seen).map((e) => e.message);
    probe.controls.push({ label: labels[i], changed: !!outcome.changed, error: outcome.error || errors[0] || null });
  }
  probe.untried = Math.max(labels.length - MAX_PRESSES, 0);
  // Errors from presses belong to the controls that raised them, not to the
  // page's load, which plain lint mode reports.
  result.consoleErrors = result.consoleErrors.slice(0, errorsBefore);
  result.probe = probe;
}

app.whenReady().then(() => {
  const targetPath = process.env.ANTON_HTML_LINT_TARGET;
  const result = { consoleErrors: [], failedRequests: [], crashed: false, crashDetails: null, emptyPage: false };

  if (!targetPath) {
    emitAndExit(result);
    return;
  }

  // "Own" scope for console errors and failed requests: the page's directory
  // for a file, the origin for a URL (the backend serves both the page and
  // its /api/*, so a failing route counts as the page's own problem).
  const isUrl = /^https?:\/\//i.test(targetPath);
  let targetDirUrl = isUrl
    ? new URL(targetPath).origin
    : pathToFileURL(path.dirname(path.resolve(targetPath))).href;
  if (!targetDirUrl.endsWith('/')) targetDirUrl += '/';

  const win = new BrowserWindow({
    show: false,
    ...(PROBE ? { width: DESKTOP.width, height: DESKTOP.height, useContentSize: true } : {}),
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

  // Over http a missing asset or an unserved route is not a network error
  // but a completed response with a 4xx/5xx status — report it the same way
  // file:// reports ERR_FILE_NOT_FOUND. Own-scope only, like the above.
  win.webContents.session.webRequest.onCompleted((details) => {
    if (details.url.startsWith(targetDirUrl) && details.statusCode >= 400) {
      result.failedRequests.push({ url: details.url, error: 'HTTP ' + details.statusCode });
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

  (isUrl ? win.loadURL(targetPath) : win.loadFile(targetPath))
    .catch(() => {})
    .finally(() => {
      setTimeout(async () => {
        // Probe mode keeps the safety net running over its presses.
        if (!PROBE) clearTimeout(hardTimeout);
        if (!result.crashed) {
          result.emptyPage = await win.webContents.executeJavaScript(EMPTY_PAGE_CHECK).catch(() => false);
        }
        if (PROBE && !result.crashed) {
          await runProbe(win, result).catch(() => {});
        }
        clearTimeout(hardTimeout);
        emitAndExit(result);
      }, SETTLE_MS);
    });
});

app.on('window-all-closed', () => {});
