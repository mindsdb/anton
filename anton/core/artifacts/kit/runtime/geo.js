/* Artifact Kit runtime: maps and globes on d3-geo with world-atlas country shapes.
   Layers: points (markers with labels), routes (great-circle lines, optional via point and flow
   animation) and regions (countries shaded by a value). Maps rotate to the data's longitude so
   routes never break at the date line; globes are orthographic, turn slowly and can be dragged. */
(function () {
  'use strict';
  var AK = window.AK;
  var RAD = Math.PI / 180;

  function layerRows(layer) { return AK.data(layer.data); }
  function num(v) { var n = AK.toNum(v); return n == null ? null : n; }
  function coord(row, latF, lonF) {
    var la = num(row[latF]), lo = num(row[lonF]);
    return la == null || lo == null ? null : [lo, la];
  }
  /* A route's end: a nested {lat, lon} (kit pages) or a pair of flat fields. */
  function end(L, row, which) {
    if (L.ends) {
      var p = row[which];
      return p && L.ends.indexOf(which) >= 0 ? coord(p, 'lat', 'lon') : null;
    }
    var la = L[which + '_lat'], lo = L[which + '_lon'];
    return la && lo ? coord(row, la, lo) : null;
  }
  function hasVia(L) { return L.ends ? L.ends.indexOf('via') >= 0 : !!L.via_lat; }
  /* Every [lon, lat] the layers draw. */
  function allCoords(layers) {
    var out = [];
    layers.forEach(function (L) {
      layerRows(L).forEach(function (r) {
        if (L.type === 'points') { var c = coord(r, L.lat || 'lat', L.lon || 'lon'); if (c) out.push(c); }
        if (L.type === 'routes') {
          ['from', 'via', 'to'].forEach(function (w) { var c = end(L, r, w); if (c) out.push(c); });
        }
      });
    });
    return out;
  }
  function meanLon(coords) {
    if (!coords.length) return 0;
    var x = 0, y = 0;
    coords.forEach(function (c) { x += Math.cos(c[0] * RAD); y += Math.sin(c[0] * RAD); });
    return Math.atan2(y, x) / RAD;
  }
  function meanLat(coords) {
    if (!coords.length) return 20;
    return coords.reduce(function (a, c) { return a + c[1]; }, 0) / coords.length;
  }
  function wrapLon(l) { while (l > 180) l -= 360; while (l < -180) l += 360; return l; }

  var ALIASES = {
    'united states': 'united states of america', usa: 'united states of america', us: 'united states of america', 'u.s.': 'united states of america',
    uk: 'united kingdom', 'great britain': 'united kingdom', russia: 'russia', 'south korea': 'south korea', 'korea, republic of': 'south korea', 'republic of korea': 'south korea',
    'north korea': 'north korea', 'czech republic': 'czechia', 'ivory coast': "côte d'ivoire", "cote d'ivoire": "côte d'ivoire", uae: 'united arab emirates',
    'democratic republic of the congo': 'dem. rep. congo', 'dr congo': 'dem. rep. congo', 'bosnia and herzegovina': 'bosnia and herz.', 'dominican republic': 'dominican rep.',
    'central african republic': 'central african rep.', 'south sudan': 's. sudan', 'equatorial guinea': 'eq. guinea', eswatini: 'eswatini', swaziland: 'eswatini', burma: 'myanmar',
    'viet nam': 'vietnam', holland: 'netherlands', 'the netherlands': 'netherlands', turkiye: 'turkey', 'türkiye': 'turkey'
  };
  function normName(s) {
    var k = String(s || '').trim().toLowerCase();
    return ALIASES[k] || k;
  }

  /* Shared drawing of layers onto a projection. Returns an object with redraw(). */
  function geoView(b, parent, mode) {
    var c = AK.card(b, parent);
    var view = AK.el('div', 'ak-view ak-geo ' + (mode === 'globe' ? 'ak-globe' : 'ak-map'));
    view.setAttribute('role', 'img');
    view.setAttribute('aria-label', b.title || (mode === 'globe' ? 'Globe' : 'Map'));
    var box = AK.el('div', 'ak-view-box');
    view.appendChild(box);
    c.body.appendChild(view);
    var legendHost = AK.el('div');
    c.body.appendChild(legendHost);
    var layers = (b.layers || []).filter(function (L) { return L && L.type; });
    Promise.all([AK.lib('d3'), AK.world(b.detail)]).then(function (res) {
      var world = res[1];
      var state = {};
      var first = true;
      AK.watchSize(view, function () {
        try {
          if (mode === 'globe') drawGlobe(b, view, box, layers, world, state);
          else drawMap(b, view, box, layers, world, state);
          if (first) { legendHost.appendChild(buildLegend(b, layers)); first = false; }
        } catch (e) { AK.fail(box, e); }
      });
    }, function (e) { AK.fail(box, e); });
    return c.card;
  }

  function colorFor(L) {
    var rows = layerRows(L);
    if (L.color && rows.length && rows[0][L.color] !== undefined) {
      var scale = AK.colorScale(AK.uniq(rows, L.color), L.colors);
      return function (r) { return scale(r[L.color]); };
    }
    var fixed = AK.resolveColor(L.color_value || L.color) || (L.type === 'routes' ? AK.css('--ak-accent') : AK.css('--ak-accent'));
    return function () { return fixed; };
  }

  function buildLegend(b, layers) {
    if (b.legend === false) return AK.el('div');
    var items = [];
    layers.forEach(function (L) {
      var rows = layerRows(L);
      var shape = L.type === 'routes' ? 'line' : (L.shape === 'square' ? '' : (L.shape === 'diamond' ? 'diamond' : 'dot'));
      if (L.color && rows.length && rows[0][L.color] !== undefined) {
        var scale = AK.colorScale(AK.uniq(rows, L.color), L.colors);
        scale.domain().forEach(function (v) { items.push({label: v, color: scale(v), shape: shape}); });
      } else if (L.name) {
        items.push({label: L.name, color: AK.resolveColor(L.color_value || L.color) || AK.css('--ak-accent'), shape: shape});
      }
      if (L.type === 'routes' && hasVia(L) && L.remaining_label) items.push({label: L.remaining_label, color: AK.css('--ak-text-3'), shape: 'dash'});
    });
    var seen = {};
    items = items.filter(function (it) { var k = it.label + '|' + it.color; if (seen[k]) return false; seen[k] = 1; return true; });
    var regions = layers.filter(function (L) { return L.type === 'regions'; })[0];
    var lg = AK.legend(items);
    if (regions) {
      var vals = layerRows(regions).map(function (r) { return num(r[regions.value]); }).filter(AK.isNum);
      if (vals.length) {
        var grad = AK.el('span', 'ak-legend-item');
        grad.appendChild(document.createTextNode(AK.fmt(Math.min.apply(null, vals), regions.format) + ' '));
        var bar = AK.el('i'); bar.style.cssText = 'width:80px;height:8px;border-radius:4px;background:linear-gradient(90deg,' + AK.css('--ak-surface-2') + ',' + (AK.resolveColor(regions.color) || AK.css('--ak-accent')) + ')';
        grad.appendChild(bar);
        grad.appendChild(document.createTextNode(' ' + AK.fmt(Math.max.apply(null, vals), regions.format) + (regions.name ? ' · ' + regions.name : '')));
        lg.appendChild(grad);
      }
    }
    return items.length || regions ? lg : AK.el('div');
  }

  function tooltipFor(L, r) {
    var title = L.label ? r[L.label] : (L.key ? r[L.key] : null);
    var fields = L.tooltip || [];
    return AK.tipRows(title == null ? '' : String(title), fields.map(function (f) {
      return [AK.label(L, f), AK.fmt(r[f], AK.fieldFormat(L, f) || 'auto')];
    }));
  }

  function symbolPath(shape, size) {
    var d3 = window.d3;
    var t = {square: d3.symbolSquare, diamond: d3.symbolDiamond, triangle: d3.symbolTriangle, star: d3.symbolStar}[shape] || d3.symbolCircle;
    return d3.symbol().type(t).size(size)();
  }

  /* Greedy label placement around each anchor; never drops a label, picks the least-overlapping spot. */
  function placeLabels(labels, width, height) {
    var placed = [];
    var base = [[10, 4, 'start'], [-10, 4, 'end'], [0, -12, 'middle'], [0, 18, 'middle'], [10, -9, 'start'], [10, 16, 'start'], [-10, -9, 'end'], [-10, 16, 'end']];
    // Nearer spots first; farther ones (drawn with a leader line) when the near ones are taken.
    var cands = base.concat(base.map(function (c) { return [c[0] * 2.4, c[1] * 2.6, c[2]]; }), base.map(function (c) { return [c[0] * 3.8, c[1] * 4.2, c[2]]; }));
    labels.forEach(function (l) { placed.push({x: l.x - 6, y: l.y - 6, w: 12, h: 12}); });
    function boxOf(l, c) {
      var w = l.w, h = 14;
      var x = c[2] === 'start' ? l.x + c[0] : (c[2] === 'end' ? l.x + c[0] - w : l.x - w / 2);
      return {x: x, y: l.y + c[1] - 11, w: w, h: h};
    }
    function overlap(a, z) {
      var ox = Math.max(0, Math.min(a.x + a.w, z.x + z.w) - Math.max(a.x, z.x));
      var oy = Math.max(0, Math.min(a.y + a.h, z.y + z.h) - Math.max(a.y, z.y));
      return ox * oy;
    }
    labels.forEach(function (l) {
      var best = null, bestScore = Infinity;
      cands.forEach(function (c, i) {
        var bx = boxOf(l, c);
        var score = i * 3;
        if (bx.x < 2 || bx.x + bx.w > width - 2 || bx.y < 2 || bx.y + bx.h > height - 2) score += 2500;
        placed.forEach(function (p) { score += overlap(bx, p) * 10; });
        if (score < bestScore) { bestScore = score; best = {c: c, box: bx}; }
      });
      l.dx = best.c[0]; l.dy = best.c[1]; l.anchor = best.c[2];
      placed.push(best.box);
    });
  }

  /* When labels still collide (a tight cluster), list them in a column beside the cluster, each
     joined to its point by a leader line. */
  function columnLabels(labels, width, height) {
    var boxes = labels.map(function (l) {
      var x = l.anchor === 'start' ? l.x + l.dx : (l.anchor === 'end' ? l.x + l.dx - l.w : l.x + l.dx - l.w / 2);
      return {x: x, y: l.y + l.dy - 11, w: l.w, h: 14};
    });
    var clash = false;
    for (var i = 0; i < boxes.length && !clash; i++) for (var j = i + 1; j < boxes.length; j++) {
      var a = boxes[i], z = boxes[j];
      if (Math.min(a.x + a.w, z.x + z.w) - Math.max(a.x, z.x) > 4 && Math.min(a.y + a.h, z.y + z.h) - Math.max(a.y, z.y) > 3) { clash = true; break; }
    }
    if (!clash || labels.length < 2) return false;
    var xs = labels.map(function (l) { return l.x; }), ys = labels.map(function (l) { return l.y; });
    var maxW = Math.max.apply(null, labels.map(function (l) { return l.w; }));
    var right = Math.max.apply(null, xs) + 34, left = Math.min.apply(null, xs) - 34;
    var useRight = right + maxW <= width - 6 || left - maxW < 6;
    var colX = useRight ? Math.min(right, width - 6 - maxW) : Math.max(left, 6 + maxW);
    var step = 17, total = (labels.length - 1) * step;
    var mid = (Math.min.apply(null, ys) + Math.max.apply(null, ys)) / 2;
    var top = Math.max(14, Math.min(height - 8 - total, mid - total / 2));
    labels.slice().sort(function (a, z) { return a.y - z.y; }).forEach(function (l, k) {
      l.anchor = useRight ? 'start' : 'end';
      l.dx = colX - l.x;
      l.dy = top + k * step + 4 - l.y;
      l.column = true;
    });
    return true;
  }

  /* Points that share a place are fanned out on screen so each stays visible. */
  function spread(items) {
    var groups = {};
    items.forEach(function (it) { var k = Math.round(it.x / 6) + ':' + Math.round(it.y / 6); (groups[k] = groups[k] || []).push(it); });
    Object.keys(groups).forEach(function (k) {
      var g = groups[k];
      if (g.length < 2) return;
      g.forEach(function (it, i) {
        var a = (i / g.length) * Math.PI * 2, rr = 7 + g.length;
        it.x += Math.cos(a) * rr; it.y += Math.sin(a) * rr;
      });
    });
  }

  /* ---------------- 2D map ---------------- */
  function drawMap(b, view, box, layers, world, st) {
    var d3 = window.d3;
    var width = Math.round(view.getBoundingClientRect().width) || 800;
    var coords = allCoords(layers);
    var lon0 = b.center_lon != null ? b.center_lon : meanLon(coords);
    var projName = b.projection || 'natural';
    var proj = (projName === 'mercator' ? d3.geoMercator() : projName === 'equal_earth' ? d3.geoEqualEarth() : d3.geoNaturalEarth1()).rotate([-lon0, 0]);
    // Extent: the data's box (padded, at least ~30 degrees) unless it spans most of the world.
    var lons = coords.map(function (c) { return wrapLon(c[0] - lon0); }), lats = coords.map(function (c) { return c[1]; });
    var spanLon = lons.length ? Math.max.apply(null, lons) - Math.min.apply(null, lons) : 360;
    var spanLat = lats.length ? Math.max.apply(null, lats) - Math.min.apply(null, lats) : 180;
    var whole = b.fit === 'world' || !coords.length || spanLon > 200 || (b.fit !== 'data' && spanLon > 150);
    var fitShape;
    var hasRoutes = layers.some(function (L) { return L.type === 'routes'; });
    if (whole) {
      // Routes may arc over the poles, so a world view with routes shows the whole sphere.
      fitShape = hasRoutes ? {type: 'Sphere'}
        : {type: 'MultiPoint', coordinates: [[lon0 - 179, 0], [lon0 + 179, 0], [lon0, 76], [lon0, -55], [lon0 - 179, 60], [lon0 + 179, -45]]};
    } else {
      var padLon = Math.max(8, (30 - spanLon) / 2, spanLon * 0.12), padLat = Math.max(6, (20 - spanLat) / 2, spanLat * 0.15);
      var lo1 = Math.min.apply(null, lons) - padLon, lo2 = Math.max.apply(null, lons) + padLon;
      var la1 = Math.max(-70, Math.min.apply(null, lats) - padLat), la2 = Math.min(82, Math.max.apply(null, lats) + padLat);
      fitShape = {type: 'MultiPoint', coordinates: [[lo1 + lon0, la1], [lo2 + lon0, la2], [lo1 + lon0, la2], [lo2 + lon0, la1]]};
    }
    var aspect = whole ? (hasRoutes ? 0.52 : 0.5) : Math.min(0.85, Math.max(0.45, (spanLat + 20) / (spanLon + 30)));
    var height = b.height ? AK.viewHeight(b, view, b.height) : Math.round(Math.max(width < 520 ? 220 : 320, Math.min(620, width * aspect)));
    box.style.height = height + 'px';
    var pad = width < 520 ? 10 : 18;
    proj.fitExtent([[pad, pad], [width - pad, height - pad]], fitShape);
    var path = d3.geoPath(proj);

    // Two stacked layers: the world (drawn once) and the marks (routes, points, labels), so motion
    // and highlights repaint only the small top layer.
    box.innerHTML = '';
    box.style.position = 'relative';
    var base = d3.select(box).append('svg').attr('class', 'ak-geo-base').attr('viewBox', '0 0 ' + width + ' ' + height).attr('width', width).attr('height', height);
    var svg = d3.select(box).append('svg').attr('class', 'ak-geo-top').attr('viewBox', '0 0 ' + width + ' ' + height).attr('width', width).attr('height', height);
    var gb = base.append('g');
    gb.append('path').datum({type: 'Sphere'}).attr('d', path).attr('fill', AK.css('--ak-water')).attr('stroke', AK.css('--ak-border'));
    gb.append('path').datum(d3.geoGraticule10()).attr('class', 'ak-graticule').attr('d', path);
    var regions = layers.filter(function (L) { return L.type === 'regions'; })[0];
    var regionVal = {};
    if (regions) layerRows(regions).forEach(function (r) { regionVal[normName(r[regions.country])] = r; });
    var rv = regions ? layerRows(regions).map(function (r) { return num(r[regions.value]); }).filter(AK.isNum) : [];
    var rmin = rv.length ? Math.min.apply(null, rv) : 0, rmax = rv.length ? Math.max.apply(null, rv) : 1;
    var rcol = regions ? d3.interpolateRgb(AK.css('--ak-surface-2'), AK.resolveColor(regions.color) || AK.css('--ak-accent')) : null;
    var regionOf = function (f) { return regions ? (regionVal[normName(f.properties.name)] || regionVal[String(f.id)] || regionVal[String(+f.id)]) : null; };
    gb.append('g').selectAll('path').data(world.countries.features).join('path')
      .attr('class', 'ak-land').attr('d', path)
      .attr('fill', function (f) {
        var r = regionOf(f);
        var v = r ? num(r[regions.value]) : null;
        return v == null ? null : rcol(rmax === rmin ? 1 : 0.15 + 0.85 * (v - rmin) / (rmax - rmin));
      });

    var g = svg.append('g');
    var transform = d3.zoomIdentity;
    g.append('rect').attr('class', 'ak-hit').attr('x', -width).attr('y', -height).attr('width', width * 3).attr('height', height * 3)
      .attr('fill', 'transparent')
      .on('click', function () { anyLink(layers, null); })
      .on('mousemove', function (e) {
        if (!regions) return;
        var m = d3.pointer(e, svg.node()), ll = proj.invert(transform.invert(m));
        var f = ll && world.countries.features.filter(function (x) { return d3.geoContains(x, ll); })[0];
        var r = f && regionOf(f);
        if (r) AK.tip(AK.tipRows(f.properties.name, [[AK.label(regions, regions.value), AK.fmt(r[regions.value], regions.format)]]), e); else AK.tip(null);
      })
      .on('mouseleave', function () { AK.tip(null); });
    var routesG = g.append('g'), ringsG = g.append('g'), pointsG = g.append('g'), labelsG = g.append('g');
    var marks = [], flows = [];
    layers.forEach(function (L) {
      if (L.type !== 'routes') return;
      var col = colorFor(L);
      layerRows(L).forEach(function (r) {
        var a = end(L, r, 'from'), z = end(L, r, 'to');
        var v = hasVia(L) ? end(L, r, 'via') : null;
        if (!a || !z) return;
        var k = L.key ? String(r[L.key]) : null;
        var segs = v ? [[a, v, false], [v, z, true]] : [[a, z, false]];
        var w = L.width ? Math.max(1.5, Math.min(7, num(r[L.width]) || 2)) : 2;
        segs.forEach(function (sg) {
          var p = routesG.append('path').datum({type: 'LineString', coordinates: [sg[0], sg[1]]})
            .attr('class', 'ak-route')
            .attr('d', path).attr('stroke', col(r))
            .attr('stroke-dasharray', sg[2] ? '3 5' : null).attr('opacity', sg[2] ? 0.55 : null)
            .attr('stroke-width', w)
            .on('mousemove', function (e) { AK.tip(tooltipFor(L, r), e); })
            .on('mouseleave', function () { AK.tip(null); })
            .on('click', function (e) { e.stopPropagation(); if (L.link && k) AK.select(L.link, k, view); });
          p.node().__w = w;
          if (L.animate && !sg[2]) flows.push({path: p.node(), color: col(r)});
          marks.push({el: p.node(), key: k, link: L.link, kind: 'route'});
        });
      });
    });
    var pts = [];
    layers.forEach(function (L) {
      if (L.type !== 'points') return;
      var rows = layerRows(L), col = colorFor(L);
      var sizes = L.size ? rows.map(function (r) { return num(r[L.size]) || 0; }) : null;
      var maxS = sizes ? Math.max.apply(null, sizes) || 1 : 1;
      rows.forEach(function (r, i) {
        var cc = coord(r, L.lat || 'lat', L.lon || 'lon');
        if (!cc) return;
        var p = proj(cc);
        if (!p) return;
        var rad = L.radius || (sizes ? 4 + 10 * Math.sqrt(sizes[i] / maxS) : (L.shape && L.shape !== 'circle' ? 6 : 5.5));
        pts.push({L: L, r: r, x: p[0], y: p[1], rad: rad, color: col(r), key: L.key ? String(r[L.key]) : null,
          label: L.label && r[L.label] != null ? String(r[L.label]) : null, sub: L.sublabel && r[L.sublabel] != null ? AK.fmt(r[L.sublabel], AK.fieldFormat(L, L.sublabel) || 'auto') : null});
      });
    });
    spread(pts);
    var showAllLabels = pts.filter(function (p) { return p.label; }).length <= (width < 520 ? 14 : 30);
    pts.forEach(function (pt) {
      var L = pt.L;
      if (L.pulse && (L.pulse === true || pt.r[L.pulse])) {
        var ring = ringsG.append('circle').attr('class', 'ak-ring').attr('cx', pt.x).attr('cy', pt.y).attr('r', pt.rad + 2).attr('stroke', pt.color);
        ring.node().__r = pt.rad + 2;
      }
      var node;
      if (L.shape && L.shape !== 'circle') {
        node = pointsG.append('path').attr('d', symbolPath(L.shape, Math.PI * pt.rad * pt.rad * 1.3)).attr('transform', 'translate(' + pt.x + ',' + pt.y + ')');
      } else {
        node = pointsG.append('circle').attr('cx', pt.x).attr('cy', pt.y).attr('r', pt.rad);
      }
      node.attr('class', 'ak-pt' + (L.shape && L.shape !== 'circle' ? ' ak-site' : '')).attr('fill', pt.color)
        .on('mousemove', function (e) { AK.tip(tooltipFor(L, pt.r), e); })
        .on('mouseleave', function () { AK.tip(null); })
        .on('click', function (e) { e.stopPropagation(); if (L.link && pt.key) AK.select(L.link, pt.key, view); });
      pt.node = node.node();
      pt.node.__pt = pt;
      marks.push({el: pt.node, key: pt.key, link: L.link, kind: 'point'});
    });
    var narrow = width < 560;
    var labels = pts.filter(function (p) { return p.label && (showAllLabels || p.L.label_always); }).map(function (p) {
      // Phones keep the sublabel (it may be the figure the page is about), in a smaller face.
      var sub = p.sub;
      var text = p.label + (sub ? ' \u00b7 ' + sub : '');
      return {p: p, x: p.x, y: p.y, w: text.length * (narrow ? 5.6 : 6.4) + 6, text: text, sub: sub};
    });
    placeLabels(labels, width, height);
    columnLabels(labels, width, height);
    var leadG = labelsG.append('g');
    labels.forEach(function (l) {
      if (l.column || Math.abs(l.dx) > 14 || Math.abs(l.dy) > 20) {
        var ex = l.x + l.dx + (l.anchor === 'end' ? 3 : (l.anchor === 'start' ? -3 : 0)), ey = l.y + l.dy - 4;
        leadG.append('line').attr('x1', l.x).attr('y1', l.y).attr('x2', ex).attr('y2', ey)
          .attr('stroke', AK.css('--ak-text-3')).attr('stroke-width', 0.8).attr('opacity', 0.75);
      }
      var t = labelsG.append('text').attr('class', 'ak-lbl').attr('x', l.x + l.dx).attr('y', l.y + l.dy).attr('text-anchor', l.anchor);
      t.append('tspan').text(l.p.label);
      if (l.sub) t.append('tspan').attr('class', 'ak-lbl-sub').text(' \u00b7 ' + l.sub);
      t.node().__l = l;
      marks.push({el: t.node(), key: l.p.key, link: l.p.L.link, kind: 'label'});
    });
    // Keep every label inside the map: measure the drawn text and nudge it in.
    labelsG.selectAll('text').each(function () {
      var bb = this.getBBox(), l = this.__l, sx = 0, sy = 0;
      if (bb.x < 4) sx = 4 - bb.x; else if (bb.x + bb.width > width - 4) sx = width - 4 - bb.x - bb.width;
      if (bb.y < 4) sy = 4 - bb.y; else if (bb.y + bb.height > height - 4) sy = height - 4 - bb.y - bb.height;
      if (sx || sy) { l.dx += sx; l.dy += sy; this.setAttribute('x', l.x + l.dx); this.setAttribute('y', l.y + l.dy); }
    });

    // Zoom and pan; marks keep their screen size.
    var zoom = d3.zoom().scaleExtent([1, 14]).translateExtent([[-width * 0.25, -height * 0.25], [width * 1.25, height * 1.25]])
      .filter(function (e) { return (e.type === 'wheel' ? e.ctrlKey || e.metaKey : (e.touches ? e.touches.length > 1 : true)) && !e.button; })
      .on('zoom', function (e) {
        transform = e.transform;
        var k = transform.k;
        gb.attr('transform', transform);
        g.attr('transform', transform);
        gb.selectAll('.ak-land').attr('stroke-width', 0.5 / k);
        routesG.selectAll('path').attr('stroke-width', function () { return this.__w / k; });
        pointsG.selectAll('circle').attr('r', function () { return this.__pt.rad / k; }).attr('stroke-width', 1.5 / k);
        pointsG.selectAll('path').attr('transform', function () { return 'translate(' + this.__pt.x + ',' + this.__pt.y + ') scale(' + (1 / k) + ')'; });
        ringsG.attr('opacity', k > 1.01 ? 0 : 1);
        labelsG.selectAll('text').attr('font-size', 11.5 / k + 'px').attr('stroke-width', 3.2 / k + 'px')
          .attr('x', function () { var l = this.__l; return l.x + l.dx / k; })
          .attr('y', function () { var l = this.__l; return l.y + l.dy / k; });
      });
    // Zoom: buttons, keyboard (+, -, 0, arrows), double-click (shift to zoom out), pinch, or
    // ctrl/cmd + wheel; drag pans. Reset returns to the whole map.
    svg.call(zoom).on('dblclick.zoom', null);
    svg.on('dblclick', function (e) {
      var p = d3.pointer(e, svg.node());
      svg.transition().duration(250).call(zoom.scaleBy, e.shiftKey ? 1 / 1.8 : 1.8, p);
    });
    svg.style('touch-action', 'pan-y');
    var zoomBy = function (f) { svg.transition().duration(250).call(zoom.scaleBy, f); };
    var reset = function () { svg.transition().duration(300).call(zoom.transform, d3.zoomIdentity); };
    var tools = AK.el('div', 'ak-geo-tools');
    [['+', function () { zoomBy(1.6); }, AK.t('zoomIn')], ['\u2212', function () { zoomBy(1 / 1.6); }, AK.t('zoomOut')],
     ['\u21ba', reset, AK.t('resetView')]].forEach(function (z) {
      var btn = AK.el('button', null, z[0]);
      btn.type = 'button'; btn.setAttribute('aria-label', z[2]); btn.title = z[2];
      btn.addEventListener('click', z[1]);
      tools.appendChild(btn);
    });
    box.appendChild(tools);
    svg.attr('tabindex', 0).on('keydown', function (e) {
      var k = e.key, step = 40;
      if (k === '+' || k === '=') zoomBy(1.6);
      else if (k === '-' || k === '_') zoomBy(1 / 1.6);
      else if (k === '0') reset();
      else if (k.indexOf('Arrow') === 0) {
        svg.transition().duration(150).call(zoom.translateBy, k === 'ArrowLeft' ? step / transform.k : k === 'ArrowRight' ? -step / transform.k : 0,
          k === 'ArrowUp' ? step / transform.k : k === 'ArrowDown' ? -step / transform.k : 0);
      } else return;
      e.preventDefault();
    });
    // Flow dots ride on a canvas above the map: it repaints alone, so motion stays cheap.
    if (flows.length) {
      var dpr = Math.min(2, window.devicePixelRatio || 1);
      var fx = document.createElement('canvas');
      fx.width = Math.round(width * dpr); fx.height = Math.round(height * dpr);
      fx.style.cssText = 'position:absolute;left:0;top:0;width:' + width + 'px;height:' + height + 'px;pointer-events:none';
      box.insertBefore(fx, tools);
      flows.forEach(function (f) { f.len = f.path.getTotalLength(); });
      animateFlows(box, fx, dpr, flows, st, function () { return transform; }, view);
    }
    bindMarks(view, marks, st);
  }

  function anyLink(layers, key) {
    layers.forEach(function (L) { if (L.link) AK.select(L.link, key, null); });
  }
  /* Linked selection: marks whose key is selected get .ak-on; the view dims the rest. */
  function bindMarks(view, marks, st) {
    var links = {};
    marks.forEach(function (m) { if (m.link) links[m.link] = true; });
    var apply = function () {
      var any = false;
      marks.forEach(function (m) {
        var sel = m.link ? AK.selected(m.link) : null;
        if (sel != null) any = true;
        m.el.classList.toggle('ak-on', sel != null && m.key === sel);
      });
      view.classList.toggle('ak-has-sel', any);
    };
    Object.keys(links).forEach(function (l) {
      if (!st['bound:' + l]) { st['bound:' + l] = true; AK.onSelect(l, function () { if (st.apply) st.apply(); }); }
    });
    st.apply = apply;
    apply();
  }

  function animateFlows(box, canvas, dpr, flows, st, getTransform, view) {
    var ctx = canvas.getContext('2d');
    st.flow = (st.flow || 0) + 1;
    var token = st.flow, t0 = performance.now(), last = 0, cost = 16;
    // Flow runs for a short while after the map appears, then rests; hovering the map resumes it.
    var until = performance.now() + 10000, sleeping = false;
    box.addEventListener('mouseenter', function () { until = performance.now() + 30000; if (sleeping) { sleeping = false; requestAnimationFrame(tick); } });
    function tick(now) {
      if (token !== st.flow || !box.isConnected) return;
      if (now > until) { sleeping = true; return; }
      if (now - last > Math.max(50, cost * 4) && box.offsetParent !== null) {
        last = now;
        var tr = getTransform(), sel = view.classList.contains('ak-has-sel');
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        if (!reduced()) flows.forEach(function (f) {
          var on = f.path.classList.contains('ak-on');
          if (sel && !on) return;
          var speed = Math.max(0.05, Math.min(0.16, 90 / Math.max(f.len, 1)));
          for (var i = 0; i < 3; i++) {
            var t = ((now - t0) / 1000 * speed + i / 3) % 1;
            var p = f.path.getPointAtLength(t * f.len), q = tr.apply([p.x, p.y]);
            ctx.globalAlpha = Math.sin(t * Math.PI) * 0.85 + 0.15;
            ctx.fillStyle = f.color;
            ctx.beginPath(); ctx.arc(q[0], q[1], on ? 3.4 : 2.7, 0, 2 * Math.PI); ctx.fill();
          }
        });
        ctx.globalAlpha = 1;
        requestAnimationFrame(function (next) { cost = cost * 0.6 + Math.min(2000, next - now) * 0.4; tick(next); });
        return;
      }
      requestAnimationFrame(tick);
    }
    requestAnimationFrame(tick);
  }

  /* ---------------- Globe ---------------- */
  /* The sphere, land, arcs and points are painted on a canvas (it repaints on its own, so a turning
     globe never slows the page); labels are real text in an overlay that follows the points. */
  function drawGlobe(b, view, box, layers, world, st) {
    var d3 = window.d3;
    var width = Math.round(view.getBoundingClientRect().width) || 800;
    var height = b.height ? AK.viewHeight(b, view, b.height) : Math.round(Math.max(300, Math.min(560, width < 520 ? width * 0.95 : width * 0.56)));
    box.style.height = height + 'px';
    var coords = allCoords(layers);
    var R = Math.min(width, height) / 2 - (width < 520 ? 14 : 26);
    var cx = width / 2, cy = height / 2;
    var center = [b.center_lon != null ? b.center_lon : meanLon(coords), b.center_lat != null ? b.center_lat : Math.max(-50, Math.min(50, meanLat(coords) * 0.7))];
    var spread0 = coords.length ? Math.max.apply(null, coords.map(function (c) { return d3.geoDistance(c, center); })) / RAD : 180;
    var mode = b.rotate === false ? 'off' : (b.rotate === 'spin' || b.rotate === true || (b.rotate !== 'sway' && spread0 > 55) ? 'spin' : 'sway');
    var speed = typeof b.speed === 'number' ? b.speed : 4;
    if (!st.rot) st.rot = [-center[0], -center[1], 0];
    st.base = [-center[0], -center[1], 0];
    var proj = d3.geoOrthographic().scale(R).translate([cx, cy]).clipAngle(90).rotate(st.rot);

    box.innerHTML = '';
    box.style.position = 'relative';
    var dpr = Math.min(2, window.devicePixelRatio || 1);
    var canvas = document.createElement('canvas');
    canvas.width = Math.round(width * dpr); canvas.height = Math.round(height * dpr);
    canvas.style.cssText = 'display:block;width:' + width + 'px;height:' + height + 'px;touch-action:pan-y;cursor:grab';
    canvas.setAttribute('role', 'img');
    canvas.setAttribute('aria-label', b.title || 'Globe');
    box.appendChild(canvas);
    var ctx = canvas.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    var path = d3.geoPath(proj, ctx);
    // Screen readers (and copy/search) get the points as text; the canvas draws them.
    var list = AK.el('ul', 'ak-sr');
    box.appendChild(list);
    var dark = document.documentElement.getAttribute('data-ak-theme') === 'dark';
    var C = {
      accent: AK.css('--ak-accent'), surface: AK.css('--ak-surface'), text: AK.css('--ak-text'),
      ocean0: dark ? '#22304d' : '#f6f9ff', ocean1: dark ? '#0b1120' : '#d3dff0',
      land: dark ? '#3d4e72' : '#c3cede', border: dark ? '#1c263b' : 'rgba(255,255,255,.9)',
      grid: dark ? 'rgba(148,163,184,.14)' : 'rgba(71,85,105,.12)', halo: AK.css('--ak-surface')
    };
    var arcs = [], pts = [];
    layers.forEach(function (L) {
      if (L.type !== 'routes') return;
      var col = colorFor(L);
      layerRows(L).forEach(function (r) {
        var a = end(L, r, 'from'), z = end(L, r, 'to');
        var v = hasVia(L) ? end(L, r, 'via') : null;
        if (!a || !z) return;
        var k = L.key ? String(r[L.key]) : null;
        (v ? [[a, v, false], [v, z, true]] : [[a, z, false]]).forEach(function (s) {
          var dist = d3.geoDistance(s[0], s[1]);
          arcs.push({L: L, r: r, a: s[0], z: s[1], rest: s[2], interp: d3.geoInterpolate(s[0], s[1]),
            h: Math.min(0.32, 0.06 + dist * 0.12), key: k, link: L.link, color: col(r), animate: L.animate !== false && !s[2]});
        });
      });
    });
    layers.forEach(function (L) {
      if (L.type !== 'points') return;
      var col = colorFor(L);
      layerRows(L).forEach(function (r) {
        var cc = coord(r, L.lat || 'lat', L.lon || 'lon');
        if (!cc) return;
        var text = L.label && r[L.label] != null ? String(r[L.label]) + (L.sublabel && r[L.sublabel] != null ? ' · ' + AK.fmt(r[L.sublabel], AK.fieldFormat(L, L.sublabel) || 'auto') : '') : null;
        var lbl = text;
        if (text) { var li = AK.el('li', null, text); list.appendChild(li); }
        pts.push({L: L, r: r, c: cc, color: col(r), key: L.key ? String(r[L.key]) : null, link: L.link, shape: L.shape,
          rad: L.radius || (L.shape && L.shape !== 'circle' ? 5.5 : 4.5), lbl: lbl});
      });
    });

    function project3(c, h) {
      var p = proj(c);
      var z = Math.cos(d3.geoDistance(c, [-st.rot[0], -st.rot[1]]));
      var dx = (p[0] - cx) * (1 + h), dy = (p[1] - cy) * (1 + h);
      return {x: cx + dx, y: cy + dy, visible: z >= 0 || Math.sqrt(dx * dx + dy * dy) > R * 1.01};
    }
    var selKey = null, now0 = performance.now();
    // The glow, ocean and shading never move: paint them once and stamp them each frame.
    function layer(paint) {
      var c = document.createElement('canvas');
      c.width = canvas.width; c.height = canvas.height;
      var x = c.getContext('2d'); x.setTransform(dpr, 0, 0, dpr, 0, 0); paint(x);
      return c;
    }
    var under = layer(function (x) {
      var glow = x.createRadialGradient(cx, cy, R * 0.9, cx, cy, R * 1.2);
      glow.addColorStop(0, hexA(C.accent, dark ? 0.35 : 0.2)); glow.addColorStop(1, hexA(C.accent, 0));
      x.fillStyle = glow; x.beginPath(); x.arc(cx, cy, R * 1.2, 0, 2 * Math.PI); x.fill();
      var ocean = x.createRadialGradient(cx - R * 0.35, cy - R * 0.4, R * 0.1, cx, cy, R * 1.05);
      ocean.addColorStop(0, C.ocean0); ocean.addColorStop(1, C.ocean1);
      x.fillStyle = ocean; x.beginPath(); x.arc(cx, cy, R, 0, 2 * Math.PI); x.fill();
    });
    var over = layer(function (x) {
      var shade = x.createRadialGradient(cx - R * 0.3, cy - R * 0.35, R * 0.5, cx, cy, R);
      shade.addColorStop(0, 'rgba(0,0,0,0)'); shade.addColorStop(1, dark ? 'rgba(0,0,0,.45)' : 'rgba(15,23,42,.16)');
      x.fillStyle = shade; x.beginPath(); x.arc(cx, cy, R, 0, 2 * Math.PI); x.fill();
    });
    var grat = d3.geoGraticule10();
    var moving = false;
    function draw(now) {
      proj.rotate(st.rot);
      ctx.setTransform(1, 0, 0, 1, 0, 0);
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.drawImage(under, 0, 0);
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.beginPath(); path(grat); ctx.strokeStyle = C.grid; ctx.lineWidth = 0.6; ctx.stroke();
      ctx.beginPath(); path(world.land); ctx.fillStyle = C.land; ctx.fill();
      if (!moving) { ctx.beginPath(); path(world.borders); ctx.strokeStyle = C.border; ctx.lineWidth = 0.6; ctx.stroke(); }
      ctx.setTransform(1, 0, 0, 1, 0, 0);
      ctx.drawImage(over, 0, 0);
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.font = '600 11.5px ' + (AK.css('--ak-font') || 'sans-serif');
      var t = (now - now0) / 1000;
      arcs.forEach(function (a) {
        var dim = selKey != null && a.key !== selKey;
        ctx.strokeStyle = hexA(a.color, dim ? 0.15 : (a.rest ? 0.55 : 0.9));
        ctx.lineWidth = a.key != null && a.key === selKey ? 3.2 : 2;
        ctx.setLineDash(a.rest ? [3, 5] : []);
        ctx.beginPath();
        var pen = false, pos = [];
        for (var i = 0; i <= 40; i++) {
          var q = project3(a.interp(i / 40), a.h * Math.sin(Math.PI * i / 40));
          pos.push(q);
          if (q.visible) { if (pen) ctx.lineTo(q.x, q.y); else ctx.moveTo(q.x, q.y); pen = true; } else pen = false;
        }
        ctx.stroke();
        ctx.setLineDash([]);
        if (a.animate && !dim && !reduced()) {
          for (var k = 0; k < 2; k++) {
            var f = ((t * 0.12 + k / 2) % 1) * 40, q = pos[Math.floor(f)];
            if (q && q.visible) { ctx.fillStyle = hexA(a.color, 0.95); ctx.beginPath(); ctx.arc(q.x, q.y, 2.6, 0, 2 * Math.PI); ctx.fill(); }
          }
        }
      });
      var c0 = [-st.rot[0], -st.rot[1]];
      var placed = [];
      pts.forEach(function (p) {
        p.vis = d3.geoDistance(p.c, c0) < Math.PI / 2 - 0.02;
        var xy = proj(p.c);
        p.x = xy[0]; p.y = xy[1];
        var on = selKey != null && p.key === selKey, dim = selKey != null && !on;
        if (p.vis) {
          ctx.globalAlpha = dim ? 0.25 : 1;
          ctx.fillStyle = p.color; ctx.strokeStyle = C.surface; ctx.lineWidth = 1.5;
          ctx.beginPath();
          var rr = on ? p.rad + 2.5 : p.rad;
          if (p.shape === 'square' || p.shape === 'diamond') {
            ctx.save(); ctx.translate(p.x, p.y); if (p.shape === 'diamond') ctx.rotate(Math.PI / 4);
            ctx.rect(-rr, -rr, rr * 2, rr * 2); ctx.restore();
          } else ctx.arc(p.x, p.y, rr, 0, 2 * Math.PI);
          ctx.fill(); ctx.stroke();
          if (on) { ctx.strokeStyle = p.color; ctx.lineWidth = 1.5; ctx.beginPath(); ctx.arc(p.x, p.y, rr + 5 + 3 * Math.sin(t * 3), 0, 2 * Math.PI); ctx.stroke(); }
          ctx.globalAlpha = 1;
        }
        if (p.lbl && p.vis) {
          var tw = ctx.measureText(p.lbl).width;
          // Right of the point unless that runs off the canvas; then left of it.
          var bx = {x: p.x + 8 + tw > width - 6 ? p.x - 8 - tw : p.x + 8, y: p.y - 8, w: tw, h: 14};
          placed.forEach(function (o) { if (bx.x < o.x + o.w && bx.x + bx.w > o.x && bx.y < o.y + o.h && bx.y + bx.h > o.y) bx.y = o.y + o.h + 1; });
          placed.push(bx);
          ctx.globalAlpha = dim ? 0.3 : 1;
          ctx.lineWidth = 3.2; ctx.strokeStyle = C.halo; ctx.lineJoin = 'round';
          ctx.strokeText(p.lbl, bx.x, bx.y + 11); ctx.fillStyle = C.text; ctx.fillText(p.lbl, bx.x, bx.y + 11);
          ctx.globalAlpha = 1;
        }
      });
    }
    draw(performance.now());

    // Motion: slow spin (or a gentle sway around regional data); paused while the user drives it.
    var last = null, pausedUntil = 0, t0 = performance.now(), onScreen = true;
    st.anim = (st.anim || 0) + 1;
    var token = st.anim;
    if (window.IntersectionObserver) new IntersectionObserver(function (en) { onScreen = en[0].isIntersecting; }).observe(box);
    // Adaptive budget: a frame that takes long to reach the screen spaces the next one out, so a
    // slow device keeps most of its time for the reader (motion stays right, just less frequent).
    var cost = 16;
    function frame(now) {
      if (token !== st.anim || !box.isConnected) return;
      if (last != null && now - last < Math.max(66, cost * 4)) { requestAnimationFrame(frame); return; }
      if (last != null && onScreen && box.offsetParent !== null) {
        moving = now > pausedUntil && !st.tween && mode !== 'off' && !reduced();
        if (moving) {
          // Real elapsed time, so the turn keeps its speed when frames are sparse (slow devices).
          var dt = Math.min(2.5, (now - last) / 1000);
          if (mode === 'spin') st.rot[0] += speed * dt;
          else st.rot[0] = st.base[0] + Math.sin((now - t0) / 1000 * 0.26) * 22;
        }
        draw(now);
        last = now;
        requestAnimationFrame(function (next) { cost = cost * 0.6 + Math.min(2000, next - now) * 0.4; frame(next); });
        return;
      }
      last = now;
      requestAnimationFrame(frame);
    }
    requestAnimationFrame(frame);
    d3.select(canvas).call(d3.drag().filter(function (e) { return !e.touches && !e.button; })
      .on('start', function () { pausedUntil = Infinity; canvas.style.cursor = 'grabbing'; })
      .on('drag', function (e) { st.rot[0] += e.dx * 0.35; st.rot[1] = Math.max(-80, Math.min(80, st.rot[1] - e.dy * 0.35)); draw(performance.now()); })
      .on('end', function () { pausedUntil = performance.now() + 4000; canvas.style.cursor = 'grab'; }));
    canvas.style.touchAction = 'pan-y';
    function nearest(e) {
      var r = canvas.getBoundingClientRect(), x = e.clientX - r.left, y = e.clientY - r.top, best = null, bd = 12;
      pts.forEach(function (p) { if (!p.vis) return; var d = Math.hypot(p.x - x, p.y - y); if (d < bd) { bd = d; best = p; } });
      return best;
    }
    canvas.addEventListener('mousemove', function (e) { var p = nearest(e); if (p) AK.tip(tooltipFor(p.L, p.r), e); else AK.tip(null); });
    canvas.addEventListener('mouseleave', function () { AK.tip(null); });
    canvas.addEventListener('click', function (e) { var p = nearest(e); if (p && p.link && p.key) AK.select(p.link, p.key, view); });

    function focus(sel) {
      var hit = pts.filter(function (p) { return p.key === sel; })[0] || null;
      var target = hit ? hit.c : null;
      if (!target) { var arc = arcs.filter(function (a) { return a.key === sel; })[0]; if (arc) target = arc.interp(0.5); }
      if (!target) return;
      var from = st.rot.slice(), to = [-target[0], -Math.max(-60, Math.min(60, target[1])), 0];
      while (to[0] - from[0] > 180) to[0] -= 360;
      while (from[0] - to[0] > 180) to[0] += 360;
      var ip = d3.interpolate(from, to), start = performance.now();
      st.tween = true; pausedUntil = performance.now() + 8000;
      function step(now) {
        var k = Math.min(1, (now - start) / 900), e = k < 0.5 ? 2 * k * k : 1 - Math.pow(-2 * k + 2, 2) / 2;
        st.rot = ip(e); draw(now);
        if (k < 1) requestAnimationFrame(step); else { st.tween = false; pausedUntil = performance.now() + 8000; }
      }
      requestAnimationFrame(step);
    }
    var links = {};
    pts.concat(arcs).forEach(function (m) { if (m.link) links[m.link] = true; });
    Object.keys(links).forEach(function (l) {
      if (!st['focus:' + l]) {
        st['focus:' + l] = true;
        AK.onSelect(l, function (sel) { if (st.select) st.select(sel); });
      }
      var cur = AK.selected(l);
      if (cur != null) selKey = cur;
    });
    st.select = function (sel) { selKey = sel; if (sel != null) focus(sel); else draw(performance.now()); };
  }
  function hexA(color, a) {
    var c = String(color).trim(), m;
    if ((m = c.match(/^#([0-9a-f]{3})$/i))) c = '#' + m[1].split('').map(function (x) { return x + x; }).join('');
    if ((m = c.match(/^#([0-9a-f]{6})/i))) {
      var n = parseInt(m[1], 16);
      return 'rgba(' + (n >> 16) + ',' + ((n >> 8) & 255) + ',' + (n & 255) + ',' + a + ')';
    }
    if ((m = c.match(/^rgba?\(([^)]+)\)/))) { var p = m[1].split(',').slice(0, 3).join(','); return 'rgba(' + p + ',' + a + ')'; }
    return c;
  }
  function reduced() { return window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches; }

  AK.blocks.map = function (b, parent) { return geoView(b, parent, 'map'); };
  AK.blocks.globe = function (b, parent) { return geoView(b, parent, 'globe'); };
})();
