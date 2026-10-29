/* Artifact Kit runtime: timelines and Gantt charts (custom SVG).
   Rows hold bar layers (start-end spans, one lane per layer so plans can be compared), milestone
   layers (dated events) and page-wide markers (dashed lines such as "as of" or a deadline).
   Hovering or tapping a milestone shows its detail in the panel under the chart. */
(function () {
  'use strict';
  var AK = window.AK;
  var DAY = 86400000;

  function t(v) { var d = AK.toDate(v); return d ? d.getTime() : null; }

  function ticks(t0, t1, width) {
    var span = (t1 - t0) / DAY, out = [], d = new Date(t0), step, fmt;
    var maxTicks = Math.max(3, Math.floor(width / 78));
    if (span <= maxTicks * 2) { step = Math.max(1, Math.ceil(span / maxTicks)); fmt = 'date_short'; d = new Date(Date.UTC(d.getUTCFullYear(), d.getUTCMonth(), d.getUTCDate())); while (d.getTime() <= t1) { if (d.getTime() >= t0) out.push(d.getTime()); d = new Date(d.getTime() + step * DAY); } }
    else if (span <= maxTicks * 7 * 1.2) { // weeks, on Mondays
      var wk = Math.max(1, Math.ceil(span / 7 / maxTicks)); fmt = 'date_short';
      d = new Date(Date.UTC(d.getUTCFullYear(), d.getUTCMonth(), d.getUTCDate()));
      while (d.getUTCDay() !== 1) d = new Date(d.getTime() + DAY);
      while (d.getTime() <= t1) { out.push(d.getTime()); d = new Date(d.getTime() + 7 * wk * DAY); }
    } else {
      var months = Math.max(1, Math.ceil(span / 30.4 / maxTicks)); fmt = 'month';
      d = new Date(Date.UTC(d.getUTCFullYear(), d.getUTCMonth() + 1, 1));
      while (d.getTime() <= t1) { out.push(d.getTime()); d = new Date(Date.UTC(d.getUTCFullYear(), d.getUTCMonth() + months, 1)); }
    }
    return {values: out, fmt: fmt};
  }

  AK.blocks.timeline = function (b, parent) {
    var c = AK.card(b, parent);
    var view = AK.el('div', 'ak-view ak-svgview ak-timeline');
    c.body.appendChild(view);
    var bars = [].concat(b.bars || []), mss = [].concat(b.milestones || []), markers = [].concat(b.markers || []);
    // Legend: one item per named layer, or per colour category.
    var items = [];
    bars.forEach(function (L) {
      var rows = AK.data(L.data);
      if (L.color && rows.length && rows[0][L.color] !== undefined) {
        var sc = AK.colorScale(AK.uniq(rows, L.color), L.colors);
        sc.domain().forEach(function (v) { items.push({label: v, color: sc(v)}); });
      } else if (L.name) items.push({label: L.name, color: AK.resolveColor(L.color_value || L.tone || L.color) || AK.css('--ak-accent')});
    });
    mss.forEach(function (L) {
      var rows = AK.data(L.data);
      if (L.color && rows.length && rows[0][L.color] !== undefined) {
        var sc = AK.colorScale(AK.uniq(rows, L.color), L.colors);
        sc.domain().forEach(function (v) { items.push({label: v, color: sc(v), shape: L.shape === 'diamond' ? 'diamond' : 'dot'}); });
      } else if (L.name) items.push({label: L.name, color: AK.resolveColor(L.color_value || L.tone || L.color) || AK.css('--ak-text-2'), shape: L.shape === 'diamond' ? 'diamond' : 'dot'});
    });
    markers.forEach(function (m) { if (m.label) items.push({label: m.label, color: AK.resolveColor(m.tone || 'neutral'), shape: 'dash'}); });
    if (items.length && b.legend !== false) c.body.insertBefore(AK.legend(items), view);
    var panel = null;
    if (mss.length && b.detail !== false) {
      panel = AK.el('div', 'ak-detail-inline');
      c.body.appendChild(panel);
    }
    var selected = null;
    AK.watchSize(view, function (w) {
      try { view.innerHTML = ''; view.appendChild(draw(w)); } catch (e) { AK.fail(view, e); }
    });

    function rowKeys() {
      var keys = [], labels = {}, subs = {};
      if (b.rows && b.rows.data) {
        AK.data(b.rows.data).forEach(function (r) {
          var k = String(r[b.rows.key]);
          if (keys.indexOf(k) < 0) keys.push(k);
          labels[k] = b.rows.label ? String(r[b.rows.label]) : k;
          if (b.rows.sublabel && r[b.rows.sublabel] != null) subs[k] = AK.fmt(r[b.rows.sublabel], AK.fieldFormat(b.rows, b.rows.sublabel) || 'auto');
        });
      }
      bars.concat(mss).forEach(function (L) {
        AK.data(L.data).forEach(function (r) {
          var k = String(r[L.row] == null ? '' : r[L.row]);
          if (keys.indexOf(k) < 0) { keys.push(k); labels[k] = k; }
        });
      });
      return {keys: keys, labels: labels, subs: subs};
    }

    function draw(width) {
      var narrow = width < 560;
      var R = rowKeys();
      var lanes = Math.max(1, bars.filter(function (L) { return !L.overlay; }).length);
      var rowH = b.row_height || (lanes > 1 ? 30 + lanes * 14 : 46);
      var longest = Math.max.apply(null, R.keys.map(function (k) { return String(R.labels[k]).length; }).concat([4]));
      var labelW = narrow ? Math.min(110, Math.max(70, longest * 6.6)) : Math.min(230, Math.max(110, longest * 7.4 + 18));
      var top = 34 + (markers.length ? 18 : 0), right = narrow ? 14 : 30;
      var H = top + R.keys.length * rowH + 10;
      var times = [];
      bars.forEach(function (L) { AK.data(L.data).forEach(function (r) { [t(r[L.start]), t(r[L.end])].forEach(function (x) { if (x != null) times.push(x); }); }); });
      mss.forEach(function (L) { AK.data(L.data).forEach(function (r) { var x = t(r[L.date]); if (x != null) times.push(x); }); });
      markers.forEach(function (m) { var x = t(m.date); if (x != null) times.push(x); });
      if (b.range) { times.push(t(b.range[0])); times.push(t(b.range[1])); }
      times = times.filter(function (x) { return x != null; });
      if (!times.length) throw new Error('timeline has no dates');
      var t0 = Math.min.apply(null, times), t1 = Math.max.apply(null, times);
      var padT = Math.max(DAY * 1.5, (t1 - t0) * 0.04);
      if (!b.range) { t0 -= padT; t1 += padT; } else { t0 = t(b.range[0]); t1 = t(b.range[1]); }
      var plotW = width - labelW - right;
      var X = function (v) { return labelW + (v - t0) / (t1 - t0) * plotW; };
      var svg = AK.svg('svg', {viewBox: '0 0 ' + width + ' ' + H, width: width, height: H, role: 'img', 'aria-label': b.title || 'Timeline'});
      var gGrid = AK.svg('g'), gRows = AK.svg('g'), gBars = AK.svg('g'), gMs = AK.svg('g'), gMark = AK.svg('g');
      [gGrid, gRows, gMark, gBars, gMs].forEach(function (g) { svg.appendChild(g); });
      var tk = ticks(t0, t1, plotW);
      tk.values.forEach(function (v) {
        var x = X(v);
        gGrid.appendChild(AK.svg('line', {x1: x, x2: x, y1: top - 6, y2: H - 6, 'class': 'ak-tl-grid'}));
        var tx = AK.svg('text', {x: x, y: top - 12, 'text-anchor': 'middle', 'class': 'ak-tl-axis'});
        tx.textContent = AK.fmt(new Date(v).toISOString().slice(0, 10), tk.fmt);
        gGrid.appendChild(tx);
      });
      var rowY = {};
      R.keys.forEach(function (k, i) {
        var y = top + i * rowH;
        rowY[k] = y;
        var g = AK.svg('g', {'class': 'ak-tl-row'});
        g.setAttribute('data-key', k);
        if (i % 2 === 0) g.appendChild(AK.svg('rect', {x: 0, y: y + 2, width: width, height: rowH - 4, rx: 8, 'class': 'ak-tl-row-bg'}));
        var lt = AK.svg('text', {x: 8, y: y + rowH / 2 + (R.subs[k] ? -1 : 4), 'class': 'ak-tl-row-label'});
        var label = String(R.labels[k]);
        var maxC = Math.floor((labelW - 12) / 6.8);
        lt.textContent = label.length > maxC ? label.slice(0, maxC - 1) + '…' : label;
        var full = AK.svg('title'); full.textContent = label; lt.appendChild(full);
        g.appendChild(lt);
        if (R.subs[k]) { var st = AK.svg('text', {x: 8, y: y + rowH / 2 + 13, 'class': 'ak-tl-row-sub'}); st.textContent = R.subs[k]; g.appendChild(st); }
        if (b.link) {
          g.style.cursor = 'pointer';
          g.addEventListener('click', function () { AK.select(b.link, k, view); });
        }
        gRows.appendChild(g);
      });
      // Bars
      var lane = 0;
      bars.forEach(function (L) {
        var rows = AK.data(L.data);
        var sc = L.color && rows.length && rows[0][L.color] !== undefined ? AK.colorScale(AK.uniq(rows, L.color), L.colors) : null;
        var fixed = AK.resolveColor(L.color_value || L.tone || (sc ? null : L.color)) || AK.palette()[lane % 10];
        var myLane = L.overlay ? Math.max(0, lane - 1) : lane;
        var bh = lanes > 1 ? 12 : 18;
        rows.forEach(function (r) {
          var k = String(r[L.row]), a = t(r[L.start]), z = t(r[L.end]);
          if (rowY[k] == null || a == null || z == null) return;
          var y = rowY[k] + (lanes > 1 ? 15 + myLane * 14 : (rowH - bh) / 2);
          var x0 = X(Math.min(a, z)), x1 = X(Math.max(a, z));
          var col = sc ? sc(r[L.color]) : fixed;
          var rect = AK.svg('rect', {x: x0, y: y, width: Math.max(3, x1 - x0), height: bh, rx: Math.min(6, bh / 2), fill: col, 'class': 'ak-tl-bar', opacity: L.opacity || (L.overlay ? 0.45 : 0.92)});
          if (L.dashed) { rect.setAttribute('fill-opacity', 0.15); rect.setAttribute('stroke', col); rect.setAttribute('stroke-dasharray', '4 3'); }
          rect.addEventListener('mousemove', function (e) {
            var pairs = [[AK.label(L, L.start), AK.fmt(r[L.start], 'date')], [AK.label(L, L.end), AK.fmt(r[L.end], 'date')]];
            (L.tooltip || []).forEach(function (f) { pairs.push([AK.label(L, f), AK.fmt(r[f], AK.fieldFormat(L, f) || 'auto')]); });
            AK.tip(AK.tipRows((L.name ? L.name + ' · ' : '') + R.labels[k], pairs), e);
          });
          rect.addEventListener('mouseleave', function () { AK.tip(null); });
          gBars.appendChild(rect);
          var text = L.label ? (r[L.label] == null ? '' : AK.fmt(r[L.label], AK.fieldFormat(L, L.label) || 'auto')) : '';
          if (text) {
            var tw = text.length * 6.4 + 12;
            var inside = (x1 - x0) > tw;
            // Outside the bar: after it when there is room, else before it, else trimmed to fit.
            var after = x1 + 5 + tw <= width - 4, before = x0 - 5 - tw >= labelW;
            var lt = AK.svg('text', {x: inside ? x0 + 6 : (after ? x1 + 5 : (before ? x0 - 5 : x0 + 6)), y: y + bh / 2 + 4,
              'font-size': 11, 'font-weight': 600, 'text-anchor': inside || after || !before ? 'start' : 'end',
              fill: inside || (!after && !before) ? '#fff' : AK.css('--ak-text-2')});
            var room = Math.floor(((x1 - x0) - 12) / 6.4);
            lt.textContent = inside || after || before || text.length <= room ? text : text.slice(0, Math.max(1, room - 1)) + '\u2026';
            gBars.appendChild(lt);
          }
        });
        if (!L.overlay) lane++;
      });
      // Milestones
      var all = [];
      mss.forEach(function (L) {
        var rows = AK.data(L.data);
        var sc = L.color && rows.length && rows[0][L.color] !== undefined ? AK.colorScale(AK.uniq(rows, L.color), L.colors) : null;
        var fixed = AK.resolveColor(L.color_value || L.tone || (sc ? null : L.color)) || AK.css('--ak-text-2');
        rows.forEach(function (r) {
          var k = String(r[L.row]), d = t(r[L.date]);
          if (rowY[k] == null || d == null) return;
          var x = X(d), y = rowY[k] + rowH / 2 + (lanes > 1 ? 4 : 0);
          var col = sc ? sc(r[L.color]) : fixed;
          var shape = L.shape || 'circle';
          var node = shape === 'diamond'
            ? AK.svg('rect', {x: x - 5.5, y: y - 5.5, width: 11, height: 11, rx: 2, transform: 'rotate(45 ' + x + ' ' + y + ')'})
            : AK.svg('circle', {cx: x, cy: y, r: L.radius || 6});
          node.setAttribute('fill', col);
          node.setAttribute('stroke', AK.css('--ak-surface'));
          node.setAttribute('stroke-width', 2);
          node.setAttribute('class', 'ak-tl-ms');
          node.setAttribute('tabindex', '0');
          var m = {L: L, r: r, k: k, node: node, x: x, y: y};
          all.push(m);
          node.addEventListener('mouseenter', function (e) { focus(m); AK.tip(tipOf(m), e); });
          node.addEventListener('mousemove', function (e) { AK.tip(tipOf(m), e); });
          node.addEventListener('mouseleave', function () { AK.tip(null); });
          node.addEventListener('click', function () { focus(m); if (b.link) AK.select(b.link, k, view); });
          node.addEventListener('focus', function () { focus(m); });
          gMs.appendChild(node);
          if (L.show_labels && r[L.label] != null) {
            var lab = AK.svg('text', {x: x, y: y - 10, 'text-anchor': 'middle', 'class': 'ak-tl-lbl'});
            lab.textContent = String(r[L.label]);
            gMs.appendChild(lab);
          }
        });
      });
      // Markers
      markers.forEach(function (mk, i) {
        var d = t(mk.date);
        if (d == null) return;
        var x = X(d), col = AK.resolveColor(mk.tone || 'neutral');
        gMark.appendChild(AK.svg('line', {x1: x, x2: x, y1: top - 4 - (markers.length ? 14 : 0), y2: H - 6, stroke: col, 'class': 'ak-tl-marker'}));
        var lt = AK.svg('text', {x: x, y: 12 + (i % 2) * 0, 'text-anchor': x > width - 90 ? 'end' : (x < labelW + 60 ? 'start' : 'middle'), fill: col, 'class': 'ak-tl-marker-lbl'});
        lt.textContent = (mk.label ? mk.label + ' · ' : '') + AK.fmt(mk.date, 'date_short');
        gMark.appendChild(lt);
      });
      function tipOf(m) {
        var L = m.L;
        var pairs = [[AK.label(L, L.date), AK.fmt(m.r[L.date], 'date')]];
        (L.tooltip || []).forEach(function (f) { pairs.push([AK.label(L, f), AK.fmt(m.r[f], AK.fieldFormat(L, f) || 'auto')]); });
        return AK.tipRows((L.label ? String(m.r[L.label]) + ' · ' : '') + R.labels[m.k], pairs);
      }
      var ordered = all.slice().sort(function (a, z) { return (t(a.r[a.L.date]) - t(z.r[z.L.date])) || (R.keys.indexOf(a.k) - R.keys.indexOf(z.k)); });
      function focus(m) {
        selected = m ? (m.k + '|' + m.r[m.L.date] + '|' + (m.L.label ? m.r[m.L.label] : '')) : null;
        all.forEach(function (x) {
          x.node.setAttribute('stroke', x === m ? AK.css('--ak-text') : AK.css('--ak-surface'));
          x.node.setAttribute('stroke-width', x === m ? 3 : 2);
        });
        if (!panel || !m) return;
        panel.innerHTML = '';
        var L = m.L;
        // Step through milestones in date order: the hover detail, for touch and keyboard too.
        var head = AK.el('div', 'ak-step');
        head.appendChild(AK.el('div', 'ak-kpi-label', AK.t('details')));
        var at = ordered.indexOf(m);
        var nav = AK.el('div', 'ak-step-nav');
        [['\u2039', -1, AK.t('prev')], ['\u203a', 1, AK.t('next')]].forEach(function (d) {
          var btn = AK.el('button', 'ak-chipbtn', d[0]);
          btn.type = 'button'; btn.setAttribute('aria-label', d[2]); btn.title = d[2];
          btn.disabled = at + d[1] < 0 || at + d[1] >= ordered.length;
          btn.addEventListener('click', function () { focus(ordered[at + d[1]]); });
          nav.appendChild(btn);
        });
        nav.insertBefore(AK.el('span', 'ak-note', (at + 1) + ' / ' + ordered.length), nav.lastChild);
        head.appendChild(nav);
        panel.appendChild(head);
        panel.appendChild(AK.el('div', 'ak-card-title', (L.label ? String(m.r[L.label]) + ' · ' : '') + R.labels[m.k]));
        var fields = (L.detail_fields || [L.date].concat(L.tooltip || []));
        panel.appendChild(AK.detailList(fields.map(function (f) { return [AK.label(L, f), AK.fmt(m.r[f], f === L.date ? 'date' : (AK.fieldFormat(L, f) || 'auto'))]; })));
      }
      if (b.link) {
        var mark = function (sel) {
          Array.prototype.forEach.call(gRows.children, function (g) { g.classList.toggle('ak-on', sel != null && g.getAttribute('data-key') === sel); });
        };
        AK.onSelect(b.link, mark);
        mark(AK.selected(b.link));
      }
      // Start with the latest milestone shown in the panel, so the detail view is never empty.
      if (all.length) {
        var keep = all.filter(function (m) { return (m.k + '|' + m.r[m.L.date] + '|' + (m.L.label ? m.r[m.L.label] : '')) === selected; })[0];
        var latest = all.slice().sort(function (a, z) { return t(z.r[z.L.date]) - t(a.r[a.L.date]); })[0];
        focus(keep || latest);
      }
      return svg;
    }
    return c.card;
  };
})();
