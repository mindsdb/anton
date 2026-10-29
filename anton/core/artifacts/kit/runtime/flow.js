/* Artifact Kit runtime: network and flow diagrams (custom SVG, no library).
   Nodes sit in tiers (given by a group order, or by longest path from the sources); order within a
   tier reduces crossings. style "graph": labelled nodes, links with arrows and value labels.
   style "sankey": bands whose width is the flow. Wide screens lay tiers left to right, phones top to
   bottom. Clicking a node highlights its links. */
(function () {
  'use strict';
  var AK = window.AK;

  function build(b) {
    var N = b.nodes || {}, Lk = b.links || {};
    var nrows = AK.data(N.data), lrows = AK.data(Lk.data);
    var idF = N.id || 'id', labelF = N.label || idF, groupF = N.group || null;
    var src = Lk.source || 'source', tgt = Lk.target || 'target', valF = Lk.value || null;
    var nodes = {}, order = [];
    function add(id, row) {
      id = String(id);
      if (!nodes[id]) { nodes[id] = {id: id, row: row || {}, label: id, group: null, ins: [], outs: [], vin: 0, vout: 0}; order.push(id); }
      if (row) {
        nodes[id].row = row;
        nodes[id].label = row[labelF] != null ? String(row[labelF]) : id;
        nodes[id].group = groupF && row[groupF] != null ? String(row[groupF]) : null;
      }
      return nodes[id];
    }
    nrows.forEach(function (r) { if (r[idF] != null) add(r[idF], r); });
    var links = [];
    lrows.forEach(function (r, i) {
      if (r[src] == null || r[tgt] == null) return;
      var s = add(r[src]), t = add(r[tgt]);
      var v = valF ? AK.toNum(r[valF]) : null;
      var l = {i: i, s: s, t: t, row: r, v: v, key: Lk.key ? String(r[Lk.key]) : s.id + '→' + t.id};
      links.push(l);
      s.outs.push(l); t.ins.push(l);
      s.vout += v || 0; t.vin += v || 0;
    });
    var list = order.map(function (id) { return nodes[id]; });
    // Tiers: explicit group order first, else longest path from sources.
    var tierOrder = b.tiers || null;
    if (tierOrder && groupF) {
      list.forEach(function (n) { var k = tierOrder.indexOf(n.group); n.tier = k >= 0 ? k : null; });
      var fallback = Math.max.apply(null, list.map(function (n) { return n.tier == null ? -1 : n.tier; })) + 1;
      list.forEach(function (n) { if (n.tier == null) n.tier = fallback; });
    } else {
      list.forEach(function (n) { n.tier = 0; });
      for (var it = 0; it < list.length + 1; it++) {
        var changed = false;
        links.forEach(function (l) { if (l.t.tier < l.s.tier + 1 && l.s.tier + 1 < list.length) { l.t.tier = l.s.tier + 1; changed = true; } });
        if (!changed) break;
      }
      // Pull sinks right so final destinations line up.
      var maxT = Math.max.apply(null, list.map(function (n) { return n.tier; }).concat([0]));
      list.forEach(function (n) { if (!n.outs.length && n.ins.length) n.tier = maxT; });
    }
    var tiers = [];
    list.forEach(function (n) { (tiers[n.tier] = tiers[n.tier] || []).push(n); });
    tiers = tiers.filter(Boolean);
    tiers.forEach(function (t, ti) { t.forEach(function (n, k) { n.tier = ti; n.pos = k; }); });
    // Barycentre sweeps
    for (var pass = 0; pass < 6; pass++) {
      var down = pass % 2 === 0;
      var seq = down ? tiers.slice(1) : tiers.slice(0, -1).reverse();
      seq.forEach(function (t) {
        t.forEach(function (n) {
          var nb = (down ? n.ins.map(function (l) { return l.s; }) : n.outs.map(function (l) { return l.t; }));
          n.bc = nb.length ? nb.reduce(function (a, m) { return a + m.pos; }, 0) / nb.length : n.pos;
        });
        t.sort(function (a, z) { return a.bc - z.bc; });
        t.forEach(function (n, k) { n.pos = k; });
      });
    }
    return {nodes: list, links: links, tiers: tiers, valF: valF};
  }

  function wrap(text, maxChars) {
    var words = String(text).split(/\s+/), lines = [], cur = '';
    words.forEach(function (w) {
      if (!cur) cur = w;
      else if ((cur + ' ' + w).length <= maxChars) cur += ' ' + w;
      else { lines.push(cur); cur = w; }
    });
    if (cur) lines.push(cur);
    if (lines.length > 2) lines = [lines[0], lines.slice(1).join(' ')];
    return lines.map(function (l) { return l.length > maxChars + 3 ? l.slice(0, maxChars + 1) + '…' : l; });
  }

  /* Narrow screens where a tier's nodes cannot sit side by side with readable labels: the same
     flows as a list, each node with its outgoing links as value bars. */
  function fitsSideBySide(G, width) {
    var maxPer = Math.max.apply(null, G.tiers.map(function (t) { return t.length; }));
    var slot = (width - 24) / Math.max(1, maxPer);
    var word = Math.max.apply(null, G.nodes.map(function (n) {
      return Math.max.apply(null, String(n.label).split(/\s+/).map(function (w) { return w.length; }));
    }).concat([1]));
    return slot >= 84 && word * 6.6 + 18 <= slot;
  }

  function flowList(b, view, G) {
    var Lk = b.links || {}, N = b.nodes || {};
    var vfmt = b.value_format || AK.fieldFormat(Lk, G.valF) || 'auto';
    var gcol = AK.colorScale(AK.uniq(G.nodes.map(function (n) { return {g: n.group}; }), 'g'), N.colors);
    var maxV = Math.max.apply(null, G.links.map(function (l) { return l.v || 0; }).concat([1]));
    var box = AK.el('div', 'ak-flowlist');
    var rows = [];
    G.tiers.forEach(function (t) {
      t.forEach(function (n) {
        var outs = G.links.filter(function (l) { return l.s === n; }).sort(function (a, z) { return (z.v || 0) - (a.v || 0); });
        if (!outs.length) return;
        var item = AK.el('div', 'ak-fl-node');
        var head = AK.el('div', 'ak-fl-head');
        head.setAttribute('role', 'button'); head.tabIndex = 0;
        head.addEventListener('click', function () { view._akFocus(n.id); });
        head.addEventListener('keydown', function (e) { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); view._akFocus(n.id); } });
        var dot = AK.el('span', 'ak-fl-dot'); dot.style.background = gcol(n.group);
        head.appendChild(dot);
        head.appendChild(AK.el('span', 'ak-fl-name', n.label));
        if (G.valF && n.vout) head.appendChild(AK.el('span', 'ak-fl-total', AK.fmt(n.vout, vfmt)));
        item.appendChild(head);
        outs.forEach(function (l) {
          var r = AK.el('div', 'ak-fl-row');
          r.appendChild(AK.el('span', 'ak-fl-to', '\u2192 ' + l.t.label));
          if (G.valF && l.v != null) {
            var bar = AK.el('span', 'ak-fl-bar'), fill = AK.el('i');
            fill.style.width = Math.max(2, 100 * l.v / maxV) + '%';
            fill.style.background = gcol(n.group);
            bar.appendChild(fill); r.appendChild(bar);
            r.appendChild(AK.el('span', 'ak-fl-val', AK.fmt(l.v, vfmt)));
          }
          item.appendChild(r);
          rows.push({el: r, l: l});
        });
        item.__node = n;
        box.appendChild(item);
      });
    });
    view._akFocus = function (id) {
      view.__sel = view.__sel === id ? null : id;
      if (view._akChips) view._akChips(view.__sel);
      rows.forEach(function (x) {
        x.el.classList.toggle('ak-dim', view.__sel != null && x.l.s.id !== view.__sel && x.l.t.id !== view.__sel);
      });
      Array.prototype.forEach.call(box.children, function (item) {
        var on = view.__sel != null && item.__node.id === view.__sel;
        item.firstChild.setAttribute('aria-pressed', String(on));
        item.classList.toggle('ak-dim', view.__sel != null && !on && !rows.some(function (x) {
          return x.el.parentNode === item && !x.el.classList.contains('ak-dim'); }));
      });
    };
    return box;
  }

  function draw(b, view, G, width) {
    if (width < 600 && !fitsSideBySide(G, width)) return flowList(b, view, G);
    var svgNS = 'http://www.w3.org/2000/svg';
    var sankey = b.style === 'sankey';
    var vertical = width < 600;
    var Lk = b.links || {}, N = b.nodes || {};
    var vfmt = b.value_format || AK.fieldFormat(Lk, G.valF) || 'auto';
    var maxV = Math.max.apply(null, G.links.map(function (l) { return l.v || 0; }).concat([1]));
    var groups = AK.uniq(G.nodes.map(function (n) { return {g: n.group}; }), 'g');
    var gcol = AK.colorScale(groups, N.colors);
    var lcolF = Lk.color || null;
    var lscale = lcolF ? AK.colorScale(AK.uniq(G.links.map(function (l) { return l.row; }), lcolF), Lk.colors, lcolF) : null;
    var linkColor = function (l) { return lscale ? lscale(l.row[lcolF]) : (sankey ? gcol(l.s.group) : AK.css('--ak-border-strong')); };
    var nT = G.tiers.length, maxPer = Math.max.apply(null, G.tiers.map(function (t) { return t.length; }));
    var labelChars = vertical ? Math.max(8, Math.floor((width - 24) / Math.max(1, maxPer) / 6.4)) : 22;
    var nodeW, nodeH, gapMain, gapCross, H, W = width;
    var M = {l: 12, r: 12, t: 16, b: 16};
    // Node boxes
    G.nodes.forEach(function (n) {
      n.lines = wrap(n.label, labelChars);
      var widest = Math.max.apply(null, n.lines.map(function (s) { return s.length; }));
      n.w = sankey ? 14 : Math.min(vertical ? (width - 24) / Math.max(1, maxPer) - 8 : 190, Math.max(56, widest * 6.9 + 22));
      n.h = sankey ? 0 : 18 + n.lines.length * 14;
      n.flow = Math.max(n.vin, n.vout, 0);
    });
    if (!vertical) {
      nodeW = 190;
      var labelSpace = sankey ? 150 : 0;
      M.l = sankey ? labelSpace : 12; M.r = sankey ? labelSpace : 12;
      var colGap = nT > 1 ? (W - M.l - M.r - (sankey ? 14 : Math.max.apply(null, G.nodes.map(function (n) { return n.w; })))) / (nT - 1) : 0;
      if (sankey) {
        var perTierFlow = G.tiers.map(function (t) { return t.reduce(function (a, n) { return a + n.flow; }, 0); });
        var Hs = b.height || Math.max(320, Math.min(620, maxPer * 64));
        var scale = (Hs - M.t - M.b - (maxPer - 1) * 14) / Math.max.apply(null, perTierFlow.concat([1]));
        G.nodes.forEach(function (n) { n.h = Math.max(4, n.flow * scale); });
        H = Hs;
        G.tiers.forEach(function (t, ti) {
          var total = t.reduce(function (a, n) { return a + n.h; }, 0) + (t.length - 1) * 14;
          var y = M.t + (H - M.t - M.b - total) / 2;
          t.forEach(function (n) { n.x = M.l + ti * colGap; n.y = y; y += n.h + 14; });
        });
        G.scale = scale;
      } else {
        gapCross = 18;
        var colH = G.tiers.map(function (t) { return t.reduce(function (a, n) { return a + n.h; }, 0) + (t.length - 1) * gapCross; });
        H = Math.max(b.height || 0, Math.max.apply(null, colH) + M.t + M.b + 10, 240);
        var maxW = Math.max.apply(null, G.nodes.map(function (n) { return n.w; }));
        // Boxes spread over the height, one slot each, so links have room between them.
        G.tiers.forEach(function (t, ti) {
          var slot = (H - M.t - M.b) / t.length;
          t.forEach(function (n, k) { n.x = M.l + ti * colGap + (maxW - n.w) / 2; n.y = M.t + slot * (k + 0.5) - n.h / 2; });
        });
      }
    } else {
      // Phones: tiers stack top to bottom.
      gapMain = sankey ? 104 : 92;
      var rowH = Math.max.apply(null, G.nodes.map(function (n) { return n.h; }).concat([sankey ? 12 : 30]));
      if (sankey) G.nodes.forEach(function (n) { n.h = 12; });
      H = M.t + M.b + nT * rowH + (nT - 1) * gapMain + (sankey ? 30 : 0);
      var scaleV = 0;
      if (sankey) {
        var perTier = G.tiers.map(function (t) { return t.reduce(function (a, n) { return a + n.flow; }, 0); });
        scaleV = (W - M.l - M.r - (maxPer - 1) * 10) / Math.max.apply(null, perTier.concat([1]));
        G.nodes.forEach(function (n) { n.w = Math.max(4, n.flow * scaleV); });
        G.scale = scaleV;
      }
      G.tiers.forEach(function (t, ti) {
        var total = t.reduce(function (a, n) { return a + n.w; }, 0) + (t.length - 1) * (sankey ? 10 : 8);
        var x = (W - total) / 2;
        // Boxes spread over the width, one slot each, so links have room between them.
        var slot = (W - M.l - M.r) / Math.max(1, t.length);
        t.forEach(function (n, k) {
          n.x = sankey ? x : M.l + slot * (k + 0.5) - n.w / 2;
          n.y = M.t + (sankey ? 30 : 0) + ti * (rowH + gapMain);
          x += n.w + (sankey ? 10 : 8);
        });
      });
    }
    var svg = document.createElementNS(svgNS, 'svg');
    svg.setAttribute('viewBox', '0 0 ' + W + ' ' + H);
    svg.setAttribute('width', W); svg.setAttribute('height', H);
    svg.setAttribute('role', 'img');
    svg.setAttribute('aria-label', b.title || 'Network diagram');
    var defs = AK.svg('defs');
    var arrows = {};
    function arrowFor(col) {
      if (!arrows[col]) {
        var id = AK.uid('arw');
        var mk = AK.svg('marker', {id: id, viewBox: '0 0 10 10', refX: 9, refY: 5, markerWidth: 6, markerHeight: 6, markerUnits: 'userSpaceOnUse', orient: 'auto'});
        mk.setAttribute('markerWidth', 9); mk.setAttribute('markerHeight', 9);
        mk.appendChild(AK.svg('path', {d: 'M0 0L10 5L0 10z', fill: col}));
        defs.appendChild(mk);
        arrows[col] = id;
      }
      return 'url(#' + arrows[col] + ')';
    }
    svg.appendChild(defs);
    var gl = AK.svg('g'), gn = AK.svg('g'), glab = AK.svg('g');
    svg.appendChild(gl); svg.appendChild(gn); svg.appendChild(glab);

    // Link end offsets so parallel links fan out at each node
    G.nodes.forEach(function (n) {
      var sortOut = n.outs.slice().sort(function (a, z) { return a.t.pos - z.t.pos; });
      var sortIn = n.ins.slice().sort(function (a, z) { return a.s.pos - z.s.pos; });
      if (sankey) {
        var o = 0; sortOut.forEach(function (l) { l.sw = Math.max(1.5, (l.v || 0) * G.scale); l.so = o + l.sw / 2; o += l.sw; });
        var q = 0; sortIn.forEach(function (l) { l.tw = Math.max(1.5, (l.v || 0) * G.scale); l.to = q + l.tw / 2; q += l.tw; });
      } else {
        sortOut.forEach(function (l, k) { l.so = (k + 1) / (sortOut.length + 1); });
        sortIn.forEach(function (l, k) { l.to = (k + 1) / (sortIn.length + 1); });
      }
    });
    var els = [];
    // Link labels keep clear of node boxes and of each other.
    var labelBoxes = G.nodes.map(function (n) {
      if (!sankey) return {x: n.x - 4, y: n.y - 4, w: n.w + 8, h: n.h + 8};
      var text = n.label + (b.show_totals !== false && n.flow ? '  ' + AK.fmt(n.flow, vfmt) : '');
      var tw = text.length * 6.9 + 8;
      if (vertical) return {x: n.x, y: n.y - 12 * Math.ceil(n.label.length * 6 / Math.max(n.w, 54)) - 10, w: Math.max(n.w, 54), h: 12 * 2 + 10};
      return {x: n.tier === 0 ? n.x - 8 - tw : n.x, y: n.y + n.h / 2 - 11, w: tw + 22, h: 22};
    });
    var hiddenLabels = 0;
    var overlap = function (a) {
      return labelBoxes.reduce(function (sum, o) {
        var w = Math.min(a.x + a.w, o.x + o.w) - Math.max(a.x, o.x), h = Math.min(a.y + a.h, o.y + o.h) - Math.max(a.y, o.y);
        return sum + (w > 0 && h > 0 ? w * h : 0);
      }, 0);
    };
    G.links.forEach(function (l) {
      var x0, y0, x1, y1;
      if (!vertical) {
        x0 = l.s.x + (sankey ? 14 : l.s.w); x1 = l.t.x;
        y0 = sankey ? l.s.y + l.so : l.s.y + l.s.h * l.so;
        y1 = sankey ? l.t.y + l.to : l.t.y + l.t.h * l.to;
      } else {
        y0 = l.s.y + (sankey ? 12 : l.s.h); y1 = l.t.y;
        x0 = sankey ? l.s.x + l.so : l.s.x + l.s.w * l.so;
        x1 = sankey ? l.t.x + l.to : l.t.x + l.t.w * l.to;
      }
      var back = (!vertical && x1 <= x0) || (vertical && y1 <= y0);
      var d;
      if (!vertical) { var mx = (x0 + x1) / 2; d = 'M' + x0 + ' ' + y0 + 'C' + mx + ' ' + y0 + ' ' + mx + ' ' + y1 + ' ' + x1 + ' ' + y1; }
      else { var my = (y0 + y1) / 2; d = 'M' + x0 + ' ' + y0 + 'C' + x0 + ' ' + my + ' ' + x1 + ' ' + my + ' ' + x1 + ' ' + y1; }
      if (back) d = 'M' + x0 + ' ' + y0 + 'C' + (x0 + 60) + ' ' + (y0 - 80) + ' ' + (x1 - 60) + ' ' + (y1 - 80) + ' ' + x1 + ' ' + y1;
      var col = linkColor(l);
      var w = sankey ? Math.max(1.5, Math.min(l.sw, l.tw)) : (G.valF && l.v != null ? 1.4 + 6 * Math.sqrt(l.v / maxV) : 1.6);
      var dashed = Lk.dashed && (Lk.dashed === true || l.row[Lk.dashed]);
      var p = AK.svg('path', {d: d, fill: 'none', stroke: col, 'stroke-width': w.toFixed(2), 'stroke-opacity': sankey ? 0.38 : 0.85,
        'stroke-dasharray': dashed ? '6 5' : null, 'marker-end': !sankey && b.arrows !== false ? arrowFor(col) : null, 'class': 'ak-link'});
      p.style.transition = 'opacity .2s ease, stroke-opacity .2s ease';
      p.addEventListener('mousemove', function (e) {
        var pairs = [[AK.label(Lk, src(Lk)), l.s.label + ' → ' + l.t.label]];
        if (G.valF) pairs.push([AK.label(Lk, G.valF), AK.fmt(l.v, vfmt)]);
        (Lk.tooltip || []).forEach(function (f) { pairs.push([AK.label(Lk, f), AK.fmt(l.row[f], AK.fieldFormat(Lk, f) || 'auto')]); });
        AK.tip(AK.tipRows(null, pairs), e);
      });
      p.addEventListener('mouseleave', function () { AK.tip(null); });
      gl.appendChild(p);
      l.el = p;
      // Value label at the curve's middle
      var lab = Lk.label === false ? null : (Lk.label ? (l.row[Lk.label] != null ? AK.fmt(l.row[Lk.label], AK.fieldFormat(Lk, Lk.label) || vfmt) : null) : (G.valF && l.v != null ? AK.fmt(l.v, vfmt) : null));
      if (lab) {
        var len = p.getTotalLength ? p.getTotalLength() : 0;
        var mid = len ? p.getPointAtLength(len * 0.5) : {x: (x0 + x1) / 2, y: (y0 + y1) / 2};
        var tw = lab.length * 6.4 + 10, th = 17;
        // Along the curve from its middle outwards, then beside it (with a leader line): the first
        // spot with no overlap, else the least.
        var bx = {x: mid.x - tw / 2, y: mid.y - th / 2, w: tw, h: th}, best = null;
        var spots = [0.5, 0.42, 0.58, 0.34, 0.66, 0.27, 0.73, 0.2, 0.8];
        [0].concat([1, -1]).some(function (side) {
          return spots.some(function (f) {
            var pt = len ? p.getPointAtLength(len * f) : mid, nx = 0, ny = 0;
            if (side && len) {
              var a = p.getPointAtLength(Math.max(0, len * f - 2)), z = p.getPointAtLength(Math.min(len, len * f + 2));
              var dx = z.x - a.x, dy = z.y - a.y, d = Math.sqrt(dx * dx + dy * dy) || 1;
              var off = (Math.abs(dx) > Math.abs(dy) ? th : tw) / 2 + 6;
              nx = -dy / d * off * side; ny = dx / d * off * side;
            }
            var cand = {x: Math.max(0, Math.min(W - tw, pt.x + nx - tw / 2)), y: Math.max(0, Math.min(H - th, pt.y + ny - th / 2)), w: tw, h: th,
              anchor: side ? pt : null};
            cand.o = overlap(cand);
            if (!best || cand.o < best.o - 1) best = cand;
            return cand.o === 0;
          });
        });
        bx = best || bx;
        // A label that cannot be placed clear of others is left to the link's tooltip, and the
        // diagram says how many values are only there.
        if (bx.o > 0.35 * tw * th) { lab = null; hiddenLabels++; }
      }
      if (lab) {
        labelBoxes.push(bx);
        var gtag = AK.svg('g', {'class': 'ak-link-label'});
        if (bx.anchor) {
          var cx = Math.max(bx.x, Math.min(bx.x + bx.w, bx.anchor.x)), cy = Math.max(bx.y, Math.min(bx.y + bx.h, bx.anchor.y));
          gtag.appendChild(AK.svg('line', {x1: bx.anchor.x, y1: bx.anchor.y, x2: cx, y2: cy, stroke: AK.css('--ak-border-strong'), 'stroke-width': 1}));
        }
        gtag.appendChild(AK.svg('rect', {x: bx.x, y: bx.y, width: bx.w, height: bx.h, rx: 8, fill: AK.css('--ak-surface'), stroke: lscale ? col : AK.css('--ak-border'), 'stroke-width': 1}));
        var tt = AK.svg('text', {x: bx.x + bx.w / 2, y: bx.y + 12.5, 'text-anchor': 'middle', 'font-size': 11, 'font-weight': 650, fill: AK.css('--ak-text')});
        tt.textContent = lab;
        gtag.appendChild(tt);
        glab.appendChild(gtag);
        l.labelEl = gtag;
      }
      els.push(l);
    });
    function src(Lkk) { return Lkk.source || 'source'; }

    G.nodes.forEach(function (n) {
      var g = AK.svg('g', {'class': 'ak-node', tabindex: 0, role: 'button'});
      g.style.cursor = 'pointer';
      g.style.transition = 'opacity .2s ease';
      var col = gcol(n.group);
      if (sankey) {
        var w = vertical ? n.w : 14, h = vertical ? 12 : n.h;
        g.appendChild(AK.svg('rect', {x: n.x, y: n.y, width: w, height: h, rx: 3, fill: col}));
        var tx, ty, anchor;
        if (!vertical) { var first = n.tier === 0; tx = first ? n.x - 8 : n.x + 22; ty = n.y + n.h / 2 + 4; anchor = first ? 'end' : 'start'; }
        else { tx = n.x + n.w / 2; ty = n.y - 8; anchor = 'middle'; }
        if (!vertical) {
          // A halo in the card's colour keeps the label readable over the bands it crosses. A label
          // wider than the room before the next column wraps onto two lines.
          var total = b.show_totals !== false && n.flow ? AK.fmt(n.flow, vfmt) : '';
          var room = n.tier === 0 ? M.l - 12 : (n.tier === nT - 1 ? W - n.x - 26 : colGap - 34);
          var one = (n.label + (total ? '  ' + total : '')).length * 6.9 <= room;
          var lines = one ? [n.label] : wrap(n.label, Math.max(8, Math.floor(room / 6.9))).slice(0, 2);
          lines.forEach(function (ln, k) {
            var t = AK.svg('text', {x: tx, y: ty + (k - (lines.length - 1) / 2) * 14, 'text-anchor': anchor, 'font-size': 12, 'font-weight': 650,
              fill: AK.css('--ak-text'), stroke: AK.css('--ak-surface'), 'stroke-width': 4, 'stroke-linejoin': 'round', 'paint-order': 'stroke'});
            t.textContent = ln;
            if (total && k === lines.length - 1) {
              var ts = AK.svg('tspan', {fill: AK.css('--ak-text-3'), 'font-weight': 500});
              ts.textContent = (one || (ln.length + total.length + 2) * 6.9 <= room) ? '  ' + total : '';
              t.appendChild(ts);
            }
            g.appendChild(t);
          });
          if (!one && total && (lines[lines.length - 1].length + total.length + 2) * 6.9 > room) {
            var t2 = AK.svg('text', {x: tx, y: ty + (lines.length + 1) / 2 * 14, 'text-anchor': anchor, 'font-size': 11.5, fill: AK.css('--ak-text-3'),
              stroke: AK.css('--ak-surface'), 'stroke-width': 4, 'stroke-linejoin': 'round', 'paint-order': 'stroke'});
            t2.textContent = total;
            g.appendChild(t2);
          }
        } else {
          // Phones: the label wraps above its node, within the node's width.
          var lines = wrap(n.label, Math.max(6, Math.floor(Math.max(n.w, 54) / 6)));
          lines.forEach(function (ln, k) {
            var tl = AK.svg('text', {x: tx, y: ty - (lines.length - 1 - k) * 12, 'text-anchor': 'middle', 'font-size': 10.5, 'font-weight': 650, fill: AK.css('--ak-text')});
            tl.textContent = ln;
            g.appendChild(tl);
          });
        }
      } else {
        var rect = AK.svg('rect', {x: n.x, y: n.y, width: n.w, height: n.h, rx: 9, fill: AK.css('--ak-surface'), stroke: col, 'stroke-width': 1.6});
        g.appendChild(rect);
        g.appendChild(AK.svg('rect', {x: n.x, y: n.y + 6, width: 4, height: n.h - 12, rx: 2, fill: col}));
        n.lines.forEach(function (line, k) {
          var tl = AK.svg('text', {x: n.x + n.w / 2 + 2, y: n.y + 15 + k * 14 + (n.h - 18 - n.lines.length * 14) / 2 + 6, 'text-anchor': 'middle', 'font-size': 12, 'font-weight': 650, fill: AK.css('--ak-text')});
          tl.textContent = line;
          g.appendChild(tl);
        });
      }
      var title = AK.svg('title'); title.textContent = n.label + (n.group ? ' (' + n.group + ')' : ''); g.appendChild(title);
      g.addEventListener('click', function (e) { e.stopPropagation(); toggle(n); });
      g.addEventListener('keydown', function (e) { if (e.key === 'Enter') toggle(n); });
      g.addEventListener('mousemove', function (e) {
        var pairs = [];
        if (n.group) pairs.push([AK.label(N, N.group), n.group]);
        if (G.valF) { if (n.vin) pairs.push(['←', AK.fmt(n.vin, vfmt)]); if (n.vout) pairs.push(['→', AK.fmt(n.vout, vfmt)]); }
        (N.tooltip || []).forEach(function (f) { pairs.push([AK.label(N, f), AK.fmt(n.row[f], AK.fieldFormat(N, f) || 'auto')]); });
        AK.tip(AK.tipRows(n.label, pairs), e);
      });
      g.addEventListener('mouseleave', function () { AK.tip(null); });
      gn.appendChild(g);
      n.el = g;
    });
    svg.addEventListener('click', function () { toggle(null); });
    var current = view.__sel ? G.nodes.filter(function (n) { return n.id === view.__sel; })[0] || null : null;
    function toggle(n) {
      current = n && current !== n ? n : null;
      view.__sel = current ? current.id : null;
      if (b.link) AK.select(b.link, view.__sel, view);
      paint();
    }
    view._akFocus = function (id) { toggle(id == null ? null : G.nodes.filter(function (n) { return n.id === id; })[0] || null); };
    function paint() {
      if (view._akChips) view._akChips(view.__sel);
      var on = {}, lon = {};
      if (current) {
        on[current.id] = 1;
        current.ins.concat(current.outs).forEach(function (l) { lon[l.i] = 1; on[l.s.id] = 1; on[l.t.id] = 1; });
      }
      G.nodes.forEach(function (n) { n.el.style.opacity = !current || on[n.id] ? 1 : 0.25; });
      G.links.forEach(function (l) {
        var hit = !current || lon[l.i];
        l.el.style.opacity = hit ? 1 : 0.12;
        if (current && hit) l.el.setAttribute('stroke-opacity', sankey ? 0.65 : 1); else l.el.setAttribute('stroke-opacity', sankey ? 0.38 : 0.85);
        if (l.labelEl) l.labelEl.style.opacity = hit ? 1 : 0.15;
      });
    }
    if (b.link) AK.onSelect(b.link, function (sel, source) {
      if (source === view) return;
      current = sel == null ? null : G.nodes.filter(function (n) { return n.id === sel; })[0] || null;
      paint();
    });
    if (b.highlight && view.__sel === undefined) { current = G.nodes.filter(function (n) { return n.id === String(b.highlight); })[0] || null; view.__sel = current ? current.id : null; }
    paint();
    svg.__hiddenLabels = hiddenLabels;
    return svg;
  }

  function legendFor(b, G) {
    var items = [];
    var N = b.nodes || {}, Lk = b.links || {};
    var groups = AK.uniq(G.nodes.map(function (n) { return {g: n.group}; }), 'g');
    var gcol = AK.colorScale(groups, N.colors);
    groups.forEach(function (g) { items.push({label: g, color: gcol(g)}); });
    if (Lk.color) {
      var vals = AK.uniq(G.links.map(function (l) { return l.row; }), Lk.color);
      var ls = AK.colorScale(vals, Lk.colors, Lk.color);
      vals.forEach(function (v) { items.push({label: (Lk.legend_prefix || '') + v, color: ls(v), shape: 'line'}); });
    }
    return items.length ? AK.legend(items) : null;
  }

  function block(b, parent) {
    var c = AK.card(b, parent);
    var view = AK.el('div', 'ak-view ak-svgview ' + (b.style === 'sankey' ? 'ak-sankey' : 'ak-network'));
    c.body.appendChild(view);
    var G = build(b);
    // Buttons that highlight one node's links (the same as clicking a node), when the page asks for them.
    var withChips = b.controls === true;
    if (withChips) {
      var bar = AK.el('div', 'ak-chips');
      bar.setAttribute('role', 'group');
      bar.appendChild(AK.el('span', 'ak-control-label', AK.t('highlight')));
      var chips = [];
      [{id: null, label: AK.t('all')}].concat(G.nodes.map(function (n) { return {id: n.id, label: n.label}; })).forEach(function (it) {
        var btn = AK.el('button', 'ak-chipbtn', it.label);
        btn.type = 'button';
        btn.addEventListener('click', function () { if (view._akFocus) view._akFocus(it.id); });
        bar.appendChild(btn);
        chips.push({id: it.id, el: btn});
      });
      view._akChips = function (sel) { chips.forEach(function (c) { c.el.setAttribute('aria-pressed', String((sel || null) === c.id)); }); };
      c.body.insertBefore(bar, view);
    }
    var lg = legendFor(b, G);
    if (lg) c.body.appendChild(lg);
    AK.watchSize(view, function (w) {
      try {
        view.innerHTML = '';
        var drawn = draw(b, view, build(b), w);
        view.appendChild(drawn);
        var n = drawn.__hiddenLabels || 0;
        if (n) view.appendChild(AK.el('div', 'ak-note', AK.t('hoverValues', {n: n})));
      } catch (e) { AK.fail(view, e); }
    });
    return c.card;
  }
  AK.blocks.network = block;
  AK.blocks.flow = function (b, parent) { return block(Object.assign({style: 'sankey'}, b), parent); };
})();
