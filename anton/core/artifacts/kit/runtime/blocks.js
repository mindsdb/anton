/* Artifact Kit runtime: layout and text blocks (section, grid, tabs, text, callout, kpis, compare,
   list, divider, html, image). */
(function () {
  'use strict';
  var AK = window.AK, B = AK.blocks;

  B.section = function (b, parent) {
    var s = AK.el('section', 'ak-section');
    if (b.title || b.subtitle) {
      var h = AK.el('div', 'ak-section-head');
      if (b.title) h.appendChild(AK.el('h2', 'ak-section-title', b.title));
      if (b.subtitle) { var p = AK.el('p', 'ak-section-sub'); p.innerHTML = AK.inline(b.subtitle); p.style.margin = '0'; h.appendChild(p); }
      s.appendChild(h);
    }
    var inner = AK.el('div', 'ak-stack');
    s.appendChild(inner);
    AK.renderAll(b.children, inner);
    parent.appendChild(s);
    return s;
  };

  B.grid = function (b, parent) {
    var cols = b.columns || 2;
    var g = AK.el('div', 'ak-grid' + (cols === 1 ? ' ak-cols-1' : ''));
    g.style.setProperty('--ak-cols', cols);
    if (b.widths && b.widths.length === cols) {
      g.style.gridTemplateColumns = b.widths.map(function (w) { return 'minmax(0,' + w + 'fr)'; }).join(' ');
      g.classList.add('ak-weighted');
    }
    (b.children || []).forEach(function (c) {
      var cell = AK.el('div', 'ak-stack');
      AK.render(c, cell);
      g.appendChild(cell);
    });
    parent.appendChild(g);
    return g;
  };
  /* weighted grids collapse to one column on phones like the others */
  var mq = window.matchMedia ? window.matchMedia('(max-width: 680px)') : null;
  function collapseWeighted() {
    document.querySelectorAll('.ak-weighted').forEach(function (g) {
      if (!g._cols) g._cols = g.style.gridTemplateColumns;
      g.style.gridTemplateColumns = mq && mq.matches ? 'minmax(0,1fr)' : g._cols;
    });
  }
  if (mq) { if (mq.addEventListener) mq.addEventListener('change', collapseWeighted); window.addEventListener('load', collapseWeighted); }

  /* Tabs: each tab holds blocks. `state` names the shared key so other tabs blocks can follow the
     same choice (e.g. one scenario switch drives two tab groups). Hidden panels stay in the page. */
  B.tabs = function (b, parent) {
    var wrap = AK.el('div', 'ak-tabs');
    var key = b.state || AK.uid('tabs');
    var tabs = b.tabs || [];
    var labels = tabs.map(function (t) { return t.label; });
    var bar = AK.el('div', b.style === 'segmented' ? 'ak-seg' : 'ak-tabs-bar');
    bar.setAttribute('role', 'tablist');
    if (b.label) bar.setAttribute('aria-label', b.label);
    var head = AK.el('div', 'ak-controls');
    if (b.label && b.style === 'segmented') head.appendChild(AK.el('span', 'ak-control-label', b.label));
    head.appendChild(bar);
    wrap.appendChild(head);
    var panels = [], buttons = [];
    tabs.forEach(function (t, i) {
      var id = AK.uid('tab');
      var btn = AK.el('button', 'ak-tab', t.label);
      btn.type = 'button';
      btn.setAttribute('role', 'tab');
      btn.id = id + '-t';
      btn.setAttribute('aria-controls', id);
      btn.addEventListener('click', function () { AK.set(key, t.label, btn); });
      btn.addEventListener('keydown', function (e) {
        if (e.key !== 'ArrowRight' && e.key !== 'ArrowLeft') return;
        var j = (i + (e.key === 'ArrowRight' ? 1 : -1) + tabs.length) % tabs.length;
        AK.set(key, labels[j]); buttons[j].focus();
      });
      bar.appendChild(btn);
      buttons.push(btn);
      var panel = AK.el('div', 'ak-tabs-panel ak-stack');
      panel.id = id;
      panel.setAttribute('role', 'tabpanel');
      panel.setAttribute('aria-labelledby', btn.id);
      wrap.appendChild(panel);
      panels.push(panel);
    });
    parent.appendChild(wrap);
    function show(value) {
      var idx = Math.max(0, labels.indexOf(value));
      buttons.forEach(function (btn, i) {
        btn.setAttribute('aria-selected', i === idx ? 'true' : 'false');
        btn.tabIndex = i === idx ? 0 : -1;
        panels[i].hidden = i !== idx;
      });
      // Views drawn in a hidden panel are sized when it opens.
      requestAnimationFrame(function () { window.dispatchEvent(new Event('resize')); });
    }
    // Panels render after they are attached, so views can measure; hidden ones render too, so
    // their text is in the page.
    tabs.forEach(function (t, i) { panels[i].hidden = false; AK.renderAll(t.children, panels[i]); });
    var initial = AK.state[key] != null && labels.indexOf(AK.state[key]) >= 0 ? AK.state[key] : labels[b.initial || 0];
    AK.on(key, show);
    show(initial);
    AK.state[key] = initial;
    return wrap;
  };

  B.text = function (b, parent) {
    var d = AK.el('div', 'ak-text' + (b.style === 'lead' ? ' ak-lead' : ''));
    d.innerHTML = AK.md(b.md || b.text || '');
    if (b.card) { var c = AK.card(b, parent); c.body.appendChild(d); return c.card; }
    parent.appendChild(d);
    return d;
  };

  B.callout = function (b, parent) {
    var tone = b.tone || 'info';
    var c = AK.el('div', 'ak-callout ak-tone-' + tone);
    var icons = {good: '✓', bad: '!', warn: '!', info: 'i', neutral: 'i', accent: '★'};
    // The icon is one glyph in a small badge: a common icon name becomes its glyph, and any other word
    // falls back to the tone's mark rather than spilling out of the badge.
    var named = {sparkles: '✨', sparkle: '✨', star: '★', lightbulb: '💡', bulb: '💡', idea: '💡', check: '✓',
      checkmark: '✓', tick: '✓', success: '✓', warning: '!', alert: '!', error: '!', info: 'i', note: 'i',
      flag: '⚑', target: '◎', trend: '↗', trendingup: '↗', trendingdown: '↘', arrowup: '↑', arrowdown: '↓', clock: '⏱',
      calendar: '📅', money: '$', dollar: '$', rocket: '🚀', fire: '🔥', heart: '♥', pin: '📍', key: '🔑'};
    var raw = b.icon == null ? '' : String(b.icon).trim();
    var glyphs = typeof Intl !== 'undefined' && Intl.Segmenter ? Array.from(new Intl.Segmenter().segment(raw)).length : Array.from(raw).length;
    var glyph = glyphs <= 2 ? raw : named[raw.toLowerCase().replace(/[^a-z]/g, '')];
    var ic = AK.el('div', 'ak-callout-icon', glyph || icons[tone] || 'i');
    ic.setAttribute('aria-hidden', 'true');
    c.appendChild(ic);
    var body = AK.el('div', 'ak-callout-body');
    if (b.title) body.appendChild(AK.el('div', 'ak-callout-title', b.title));
    var t = AK.el('div', 'ak-text');
    t.innerHTML = AK.md(b.md || b.text || '');
    body.appendChild(t);
    c.appendChild(body);
    parent.appendChild(c);
    return c;
  };

  function deltaTone(delta, better) {
    if (!AK.isNum(delta) || delta === 0 || !better) return 'neutral';
    var up = delta > 0;
    return (better === 'up' || better === 'higher') === up ? 'good' : 'bad';
  }
  function spark(values, color) {
    var w = 120, h = 34, n = values.length;
    var nums = values.filter(AK.isNum);
    if (n < 2 || !nums.length) return null;
    var lo = Math.min.apply(null, nums), hi = Math.max.apply(null, nums), span = hi - lo || 1;
    var pts = values.map(function (v, i) { return [i / (n - 1) * w, h - 3 - ((AK.isNum(v) ? v : lo) - lo) / span * (h - 6)]; });
    var svg = AK.svg('svg', {viewBox: '0 0 ' + w + ' ' + h, preserveAspectRatio: 'none', 'class': 'ak-spark', role: 'img', 'aria-hidden': 'true'});
    svg.style.width = '100%';
    var d = pts.map(function (p, i) { return (i ? 'L' : 'M') + p[0].toFixed(1) + ' ' + p[1].toFixed(1); }).join('');
    svg.appendChild(AK.svg('path', {d: d + 'L' + w + ' ' + h + 'L0 ' + h + 'Z', fill: color, opacity: 0.1}));
    svg.appendChild(AK.svg('path', {d: d, fill: 'none', stroke: color, 'stroke-width': 1.8, 'vector-effect': 'non-scaling-stroke'}));
    return svg;
  }
  B.kpis = function (b, parent) {
    var g = AK.el('div', 'ak-kpis');
    if (b.columns) g.style.gridTemplateColumns = 'repeat(auto-fit, minmax(' + Math.max(150, Math.floor(1100 / b.columns) - 24) + 'px, 1fr))';
    (b.items || []).forEach(function (k) {
      var card = AK.el('div', 'ak-card ak-kpi' + (k.tone ? ' ak-tone-' + k.tone : ''));
      card.appendChild(AK.el('div', 'ak-kpi-label', k.label));
      var val = AK.el('div', 'ak-kpi-value' + (k.tone && k.color_value ? ' ak-tone-' + k.tone : ''));
      val.textContent = k.value == null ? '\u2014' : typeof k.value === 'string' && !k.format ? k.value : AK.fmt(k.value, k.format || 'auto');
      if (val.textContent.length + (k.unit ? k.unit.length + 1 : 0) > 11) val.classList.add('ak-long');
      if (k.unit && k.value != null) val.appendChild(AK.el('span', 'ak-kpi-unit', (/^[%°]/.test(k.unit) ? '' : '\u00a0') + k.unit));
      card.appendChild(val);
      if (k.delta != null) {
        var dv = k.delta, tone = k.delta_tone || deltaTone(AK.toNum(dv), k.better);
        var dtext = typeof dv === 'string' ? dv : ((dv > 0 ? '▲ ' : dv < 0 ? '▼ ' : '') + AK.fmt(Math.abs(dv), k.delta_format || k.format || 'auto'));
        var dl = AK.el('span', 'ak-delta ak-tone-' + tone, dtext + (k.delta_label ? ' ' + k.delta_label : ''));
        card.appendChild(dl);
      }
      if (k.sub) { var s = AK.el('div', 'ak-kpi-sub'); s.innerHTML = AK.inline(k.sub); card.appendChild(s); }
      if (k.spark && k.spark.length > 1) {
        var sp = spark(k.spark, AK.toneColor(k.tone) || AK.css('--ak-accent'));
        if (sp) card.appendChild(sp);
      }
      g.appendChild(card);
    });
    parent.appendChild(g);
    return g;
  };

  /* Compare: metrics (rows) across options (columns), with the change from the first column. */
  B.compare = function (b, parent) {
    // On a narrow screen each metric becomes a card listing its values, change included.
    var c = AK.card(b, parent, 'ak-flush ak-cards-mobile');
    var wrap = AK.el('div', 'ak-table-wrap');
    wrap.style.border = '0'; wrap.style.borderRadius = '0';
    var t = AK.el('table', 'ak-table ak-compare');
    var cols = b.columns || [];
    var showDiff = b.show_diff !== false && cols.length === 2;
    var thead = AK.el('thead'), hr = AK.el('tr');
    hr.appendChild(AK.el('th', null, b.metric_label || ''));
    cols.forEach(function (col) { hr.appendChild(AK.el('th', null, col)); });
    if (showDiff) hr.appendChild(AK.el('th', null, b.diff_label || AK.t('diff')));
    thead.appendChild(hr); t.appendChild(thead);
    var tb = AK.el('tbody');
    (b.rows || []).forEach(function (r) {
      var tr = AK.el('tr');
      var th = AK.el('td');
      th.textContent = r.label + (r.unit ? ' (' + r.unit + ')' : '');
      if (r.note) { var n = AK.el('div', 'ak-note', r.note); th.appendChild(n); }
      tr.appendChild(th);
      var vals = r.values || [];
      var nums = vals.map(AK.toNum);
      var best = null;
      if (r.better && nums.filter(AK.isNum).length > 1) {
        var pick = (r.better === 'higher' || r.better === 'up') ? Math.max : Math.min;
        best = pick.apply(null, nums.filter(AK.isNum));
      }
      vals.forEach(function (v, i) {
        var td = AK.el('td', AK.isNum(nums[i]) && nums[i] === best ? 'ak-best' : null);
        td.textContent = v == null ? '—' : (typeof v === 'string' && !r.format ? v : AK.fmt(v, r.format || 'auto'));
        if (r.tones && r.tones[i]) { var txt = td.textContent; td.textContent = ''; td.appendChild(AK.el('span', 'ak-pill ak-tone-' + r.tones[i], txt)); }
        td.setAttribute('data-label', cols[i] || '');
        tr.appendChild(td);
      });
      if (showDiff) {
        var td = AK.el('td');
        if (AK.isNum(nums[0]) && AK.isNum(nums[1])) {
          var d = nums[1] - nums[0];
          var tone = !r.better || d === 0 ? 'neutral' : (((r.better === 'higher' || r.better === 'up') === d > 0) ? 'good' : 'bad');
          // A change between percentages is in points, not percent.
          var pct = !r.diff_format && /^(percent|fraction)/.test(String(r.format || ''));
          var mag = pct ? AK.fmt(Math.abs(String(r.format).indexOf('fraction') === 0 ? d * 100 : d), 'number:1') + '\u00a0' + AK.t('pts')
            : AK.fmt(Math.abs(d), r.diff_format || r.format || 'auto');
          var txt = d === 0 ? '0' : (d > 0 ? '+' : '\u2212') + mag;
          if (r.show_pct && nums[0]) txt += ' (' + (d > 0 ? '+' : '−') + AK.fmt(Math.abs(d / nums[0] * 100), 'percent:1') + ')';
          td.appendChild(AK.el('span', 'ak-delta ak-tone-' + tone, txt));
        } else td.textContent = '—';
        td.setAttribute('data-label', b.diff_label || AK.t('diff'));
        tr.appendChild(td);
      }
      tb.appendChild(tr);
    });
    t.appendChild(tb);
    wrap.appendChild(t);
    c.body.appendChild(wrap);
    return c.card;
  };

  B.list = function (b, parent) {
    var c = AK.card(b, parent);
    var ol = AK.el(b.ordered ? 'ol' : 'ul', 'ak-text');
    ol.style.margin = '0'; ol.style.paddingLeft = '20px';
    (b.items || []).forEach(function (it) { var li = AK.el('li'); li.innerHTML = AK.inline(it); ol.appendChild(li); });
    c.body.appendChild(ol);
    return c.card;
  };

  B.divider = function (b, parent) {
    var hr = AK.el('hr');
    hr.style.cssText = 'border:0;border-top:1px solid var(--ak-border);margin:6px 0;width:100%';
    parent.appendChild(hr);
    return hr;
  };

  B.image = function (b, parent) {
    var c = b.card === false ? {card: AK.el('figure'), body: null} : AK.card(b, parent);
    var img = AK.el('img');
    img.src = b.src; img.alt = b.alt || b.title || '';
    img.style.cssText = 'display:block;max-width:100%;height:auto;border-radius:8px;margin:0 auto';
    (c.body || c.card).appendChild(img);
    if (b.card === false) parent.appendChild(c.card);
    return c.card;
  };

  /* Custom HTML: the markup is the author's; its script was emitted at build time as
     AK.custom[id] = function (el, AK) {...}, since the preview policy forbids eval. */
  B.html = function (b, parent) {
    var holder = b.card === false ? null : AK.card(b, parent);
    var box = AK.el('div', 'ak-custom');
    if (b.view) box.className += ' ak-view ak-' + b.view;
    box.innerHTML = b.html || '';
    (holder ? holder.body : parent).appendChild(box);
    var fn = AK.custom[b.id];
    if (typeof fn === 'function') {
      var run = function () { try { fn(box, AK); } catch (e) { AK.fail(box, e); } };
      if (b.libs && b.libs.length) Promise.all(b.libs.map(AK.lib)).then(run, function (e) { AK.fail(box, e); });
      else run();
    }
    return holder ? holder.card : box;
  };
})();
