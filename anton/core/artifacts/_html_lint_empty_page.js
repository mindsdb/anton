// Shared by _html_lint_runner.js (Electron) and _html_lint_runner.py (Playwright).
// Heuristic, not authoritative: true only when the body has neither visible
// text nor a non-script/style/meta child element. Catches "nothing rendered
// at all" (e.g. a page whose entire markup got swallowed as text by an
// unclosed <title>/<script>/<style> — real case found in testing) without
// flaging a legitimately content-free page that fills in later (async data,
// a splash screen). False negatives are expected (e.g. a blank <canvas>).
(() => {
  if (!document.body) return true;
  const NON_VISUAL = new Set(['SCRIPT', 'STYLE', 'TEMPLATE', 'LINK', 'META', 'NOSCRIPT', 'TITLE']);
  const visibleChildren = Array.from(document.body.children).filter((el) => !NON_VISUAL.has(el.tagName));
  const hasText = document.body.innerText.trim().length > 0;
  return visibleChildren.length === 0 && !hasText;
})()
