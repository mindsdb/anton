/* Artifact Kit runtime: charts on ECharts (SVG renderer, so labels are real text).
   Kinds: bar, hbar, line, area, combo, scatter, pie, donut, heatmap, waterfall; `echarts` passes an
   option through for anything else. */
(function () {
  'use strict';
  var AK = window.AK;

  function theme() {
    return {
      text: AK.css('--ak-text'), text2: AK.css('--ak-text-2'), text3: AK.css('--ak-text-3'),
      grid: AK.css('--ak-grid'), border: AK.css('--ak-border'), surface: AK.css('--ak-surface'),
      font: AK.css('--ak-font') || 'sans-serif'
    };
  }
  function baseOption(T) {
    return {
      animationDuration: 650, animationEasing: 'cubicOut',
      textStyle: {fontFamily: T.font, color: T.text2},
      tooltip: {backgroundColor: T.surface, borderColor: T.border, textStyle: {color: T.text, fontSize: 12}, confine: true,
        extraCssText: 'border-radius:10px;box-shadow:0 10px 30px rgba(16,24,40,.16);'},
      grid: {left: 8, right: 16, top: 36, bottom: 8, containLabel: true}
    };
  }
  AK.chartTheme = theme;

  function mountChart(b, parent, kind, build) {
    var c = AK.card(b, parent);
    var view = AK.el('div', 'ak-view ak-chart ak-chart-' + kind);
    var box = AK.el('div', 'ak-view-box');
    view.appendChild(box);
    c.body.appendChild(view);
    var chart = null;
    var draw = function () {
      box.style.height = AK.viewHeight(b, view, b.kind === 'pie' || b.kind === 'donut' ? 320 : 340) + 'px';
      if (!chart) chart = window.echarts.init(box, null, {renderer: 'svg'});
      else chart.resize();
      var opt = build(view.getBoundingClientRect().width, chart);
      if (opt._akHeight) {
        if (Math.abs(opt._akHeight - box.getBoundingClientRect().height) > 2) { box.style.height = opt._akHeight + 'px'; chart.resize(); }
        delete opt._akHeight;
      }
      chart.setOption(opt, true);
      if (b.link && chart._akBind !== true) { chart._akBind = true; bindSelection(b, chart); }
    };
    AK.lib('echarts').then(function () { AK.watchSize(view, function () { try { draw(); } catch (e) { AK.fail(box, e); } }); },
      function (e) { AK.fail(box, e); });
    if (b.show_data !== false && b.data && b.kind !== 'heatmap') c.body.appendChild(dataTable(b));
    return c.card;
  }

  /* A collapsed table of the chart's numbers: readable without the picture. */
  function dataTable(b) {
    var rows = AK.data(b.data);
    var fields = [b.x].concat(b.series ? [b.series] : []).concat(ys(b)).concat(b.y2 ? [].concat(b.y2) : []).filter(Boolean);
    fields = fields.filter(function (f, i) { return fields.indexOf(f) === i; });
    var d = AK.el('details', 'ak-data');
    d.style.marginTop = '8px';
    var s = AK.el('summary', 'ak-note', AK.t('data'));
    s.style.cursor = 'pointer';
    d.appendChild(s);
    // Built when opened: a closed chart's numbers add nothing to the page until asked for.
    d.addEventListener('toggle', function () { if (d.open && !d._built) { d._built = true; d.appendChild(build()); } });
    return d;
    function build() {
    var wrap = AK.el('div', 'ak-table-wrap');
    wrap.style.marginTop = '8px';
    var t = AK.el('table', 'ak-table');
    var hr = AK.el('tr');
    fields.forEach(function (f) { hr.appendChild(AK.el('th', null, AK.label(b, f))); });
    var th = AK.el('thead'); th.appendChild(hr); t.appendChild(th);
    var tb = AK.el('tbody');
    rows.slice(0, 400).forEach(function (r) {
      var tr = AK.el('tr');
      fields.forEach(function (f) { tr.appendChild(AK.el('td', null, AK.fmt(r[f], fmtOf(b, f)))); });
      tb.appendChild(tr);
    });
    t.appendChild(tb); wrap.appendChild(t);
    return wrap;
    }
  }

  function ys(b) { return b.y == null ? [] : [].concat(b.y); }
  function fmtOf(b, f) {
    if (f === b.x) return b.x_format || AK.fieldFormat(b, f) || 'auto';
    if (b.y2 && [].concat(b.y2).indexOf(f) >= 0) return b.y2_format || AK.fieldFormat(b, f) || 'auto';
    return AK.fieldFormat(b, f) || b.format || 'auto';
  }
  function isDateField(rows, f) {
    var vals = rows.map(function (r) { return r[f]; }).filter(function (v) { return v != null; });
    return vals.length && vals.every(function (v) { return typeof v === 'string' && /^\d{4}-\d{2}-\d{2}/.test(v); });
  }

  /* Rows -> {categories, series:[{name, field, values}]}; long data pivots on `series`. */
  function shape(b, rows) {
    var cats = AK.uniq(rows, b.x);
    if (b.sort === 'desc' || b.sort === 'asc') {
      var f0 = ys(b)[0];
      var tot = {};
      rows.forEach(function (r) { var k = String(r[b.x]); tot[k] = (tot[k] || 0) + (AK.toNum(r[f0]) || 0); });
      cats.sort(function (a, z) { return (tot[String(a)] - tot[String(z)]) * (b.sort === 'desc' ? -1 : 1); });
    }
    var idx = {}; cats.forEach(function (c, i) { idx[String(c)] = i; });
    var out = [];
    if (b.series) {
      var yf = ys(b)[0];
      AK.uniq(rows, b.series).forEach(function (s) {
        var vals = cats.map(function () { return null; });
        rows.forEach(function (r) {
          if (String(r[b.series]) !== String(s)) return;
          var v = AK.toNum(r[yf]); var i = idx[String(r[b.x])];
          if (v != null) vals[i] = (vals[i] || 0) + v;
        });
        out.push({name: String(s), field: yf, values: vals});
      });
    } else {
      ys(b).forEach(function (yf) {
        var vals = cats.map(function () { return null; });
        rows.forEach(function (r) { var v = AK.toNum(r[yf]); var i = idx[String(r[b.x])]; if (v != null) vals[i] = (vals[i] || 0) + v; });
        out.push({name: AK.label(b, yf), field: yf, values: vals});
      });
    }
    return {cats: cats, series: out};
  }

  function refLines(b, T) {
    var lines = (b.ref_lines || []).slice();
    if (b.threshold && b.threshold.value != null) lines.push({value: b.threshold.value, label: b.threshold.label, tone: b.threshold.tone_above || b.threshold.tone || 'bad'});
    if (!lines.length) return null;
    return {
      silent: true, symbol: 'none',
      data: lines.map(function (l) {
        var col = AK.resolveColor(l.tone || 'neutral') || T.text3;
        var item = {lineStyle: {color: col, type: 'dashed', width: 1.5},
          label: {formatter: l.label || String(l.value), color: col, fontWeight: 600, fontSize: 11, position: 'insideEndTop'}};
        if (b.horizontal || b.kind === 'hbar') item.xAxis = l.value; else item.yAxis = l.value;
        return item;
      })
    };
  }
  function pointColors(b, rows, cats, base) {
    // Per-point colours: a status field, or a threshold the value crosses.
    if (!b.point_tones && !(b.threshold && b.threshold.value != null)) return null;
    var yf = ys(b)[0];
    return cats.map(function (c) {
      var r = rows.filter(function (x) { return String(x[b.x]) === String(c); })[0] || {};
      if (b.point_tones) {
        var tone = AK.toneOf(r[b.point_tones.field || b.point_tones], b.point_tones.map || b.point_tones.tones, b.point_tones.field || b.point_tones);
        return tone ? AK.toneColor(tone) : base;
      }
      var v = AK.toNum(r[yf]);
      var over = v != null && v > b.threshold.value;
      var t = over ? (b.threshold.tone_above || 'bad') : b.threshold.tone_below;
      return t ? AK.toneColor(t) : base;
    });
  }

  function cartesian(b, width, chart) {
    var T = theme();
    var rows = AK.data(b.data);
    var kind = b.kind || 'bar';
    var horizontal = kind === 'hbar' || b.horizontal === true;
    var s = shape(b, rows);
    var pal = AK.palette();
    var colors = b.colors || {};
    var dateX = isDateField(rows, b.x);
    var xfmt = b.x_format || AK.fieldFormat(b, b.x) || (dateX ? AK.dateFormatFor(rows.map(function (r) { return r[b.x]; }), true) : 'auto');
    var yfmt = AK.fieldFormat(b, ys(b)[0]) || b.format || 'auto';
    var axisY = AK.axisFmt(yfmt);
    var narrow = width < 520;
    var nPoints = s.cats.length * s.series.length;
    // Category names too long to sit under their bars read better as horizontal bars, whole.
    var catTexts = s.cats.map(function (c) { return String(AK.fmt(c, xfmt)); });
    var longestCat = Math.max.apply(null, catTexts.map(function (t) { return t.length; }).concat([1]));
    if (!horizontal && kind === 'bar' && !dateX && b.horizontal !== false && s.cats.length <= 30 &&
        longestCat * 6.4 > (width - 70) / Math.max(1, s.cats.length) * 1.5) horizontal = true;
    // Value labels only where they fit: a vertical bar's slot must be wider than its label.
    var labelFmt = function (v) { return AK.fmt(v, (yfmt.indexOf('currency') === 0 || Math.abs(v) >= 1e5) ? (yfmt.indexOf('currency') === 0 ? 'currency_compact' : 'compact') : yfmt); };
    var widest = Math.max.apply(null, s.series.map(function (ser) { return Math.max.apply(null, ser.values.map(function (v) { return v == null ? 0 : labelFmt(v).length; }).concat([0])); }).concat([0]));
    var slot = (width - 70) / Math.max(1, s.cats.length) / (b.stack ? 1 : Math.max(1, s.series.length));
    var fits = horizontal ? true : widest * 6.6 + 4 < slot;
    var showValues = b.show_values != null ? b.show_values : ((kind === 'bar' || kind === 'hbar') && nPoints <= (narrow ? 10 : 24) && !b.stack && fits);
    var opt = baseOption(T);
    opt.color = pal;
    // Every category keeps its label when there are few of them: long ones are shortened, not dropped.
    var fewCats = s.cats.length <= (narrow ? 10 : 16);
    // Horizontal bars wrap a long name onto lines within a share of the width (three at most).
    var labelW = Math.round(narrow ? width * 0.38 : Math.min(260, width * 0.3));
    var maxChars = horizontal ? Math.floor(labelW / 6.4) * 3 : Math.max(6, Math.floor((width - 80) / Math.max(1, s.cats.length) / 6.2));
    var catLabel = function (v) { var t = String(v); return fewCats && t.length > maxChars + 1 ? t.slice(0, maxChars) + '\u2026' : t; };
    var catAxis = {
      type: 'category', data: s.cats.map(function (c) { return AK.fmt(c, xfmt); }),
      axisLine: {lineStyle: {color: T.border}}, axisTick: {show: false},
      axisLabel: {color: T.text2, fontSize: 11, hideOverlap: !fewCats, interval: fewCats ? 0 : 'auto', formatter: catLabel,
        rotate: (!horizontal && narrow && s.cats.length > 5) ? 40 : 0,
        width: horizontal ? labelW : undefined, overflow: horizontal ? 'break' : undefined, lineHeight: 13},
      name: b.x_label || '', nameLocation: 'middle', nameGap: 28, nameTextStyle: {color: T.text3}
    };
    // Counts get whole-number ticks only (no "0, 0, 1, 1" axes).
    var allInts = s.series.every(function (ser) { return ser.values.every(function (v) { return v == null || Math.round(v) === v; }); });
    var valAxis = {
      type: 'value', min: b.y_min, max: b.y_max, splitLine: {lineStyle: {color: T.grid}}, minInterval: allInts ? 1 : undefined,
      axisLabel: {color: T.text3, fontSize: 11, formatter: axisY, hideOverlap: true},
      splitNumber: horizontal && narrow ? 3 : undefined,
      name: b.y_label || '', nameTextStyle: {color: T.text3, align: horizontal ? 'center' : 'left'}, nameGap: horizontal ? 26 : 12
    };
    if (horizontal) { catAxis.inverse = true; catAxis.nameLocation = 'end'; opt.xAxis = valAxis; opt.yAxis = catAxis; }
    else { opt.xAxis = catAxis; opt.yAxis = [valAxis]; }
    var y2 = b.y2 ? [].concat(b.y2) : [];
    // A second axis only for a measure in other units; the same units share the first axis, so the
    // line and the bars are read against one scale.
    var y2fmt0 = b.y2_format || (y2.length ? AK.fieldFormat(b, y2[0]) : null);
    var ownAxis = y2.length && !horizontal && (b.y2_axis === true || (b.y2_axis !== false && y2fmt0 && String(y2fmt0).split(':')[0] !== String(yfmt).split(':')[0]));
    if (ownAxis) {
      var y2fmt = b.y2_format || AK.fieldFormat(b, y2[0]) || 'auto';
      opt.yAxis.push({type: 'value', splitLine: {show: false}, axisLabel: {color: T.text3, fontSize: 11, formatter: AK.axisFmt(y2fmt)},
        name: b.y2_label || '', nameTextStyle: {color: T.text3}, min: b.y2_min, max: b.y2_max});
    }
    var colorsAt = pointColors(b, rows, s.cats, null);
    // Several series take the page's colour for each name, so a scenario looks the same in every chart.
    var serColor = s.series.length > 1 ? AK.colorScale(s.series.map(function (ser) { return ser.name; }), null) : null;
    opt.series = s.series.map(function (ser, i) {
      var type = (kind === 'line' || kind === 'area') ? 'line' : 'bar';
      var col = AK.resolveColor(colors[ser.name] || colors[ser.field]) || (serColor ? serColor(ser.name) : pal[i % pal.length]);
      var item = {
        name: ser.name, type: type,
        data: ser.values.map(function (v, j) {
          var c2 = colorsAt && colorsAt[j];
          return c2 ? {value: v, itemStyle: {color: c2}} : v;
        }),
        itemStyle: {color: col, borderRadius: type === 'bar' ? (horizontal ? [0, 4, 4, 0] : [4, 4, 0, 0]) : 0},
        emphasis: {focus: 'series'},
        barMaxWidth: 46
      };
      if (b.stack) { item.stack = 'total'; item.itemStyle.borderRadius = 0; }
      if (type === 'line') {
        item.smooth = b.smooth !== false ? 0.25 : false;
        item.symbol = 'circle'; item.symbolSize = s.cats.length > 30 ? 0 : 6;
        item.lineStyle = {width: 2.4, color: col};
        if (kind === 'area' || b.area) item.areaStyle = {opacity: s.series.length > 1 ? 0.12 : 0.18, color: col};
        if (b.dashed && [].concat(b.dashed).indexOf(ser.field) >= 0) item.lineStyle.type = 'dashed';
      }
      if (showValues) {
        item.label = {show: true, position: horizontal ? 'right' : 'top', color: T.text2, fontSize: 11, fontWeight: 600,
          formatter: function (p) { return labelFmt(p.value && p.value.value != null ? p.value.value : p.value); }};
        // Dense series keep only the value labels that have room; the rest are in the tooltip and the data table.
        item.labelLayout = {hideOverlap: true};
      }
      if (i === 0) { var ml = refLines(b, T); if (ml) item.markLine = ml; }
      return item;
    });
    y2.forEach(function (f, k) {
      var vals = s.cats.map(function (c) {
        var r = rows.filter(function (x) { return String(x[b.x]) === String(c); })[0];
        return r ? AK.toNum(r[f]) : null;
      });
      var col = AK.resolveColor(colors[f]) || pal[(s.series.length + k) % pal.length];
      var y2kind = b.y2_kind || 'line';
      opt.series.push({name: AK.label(b, f), type: y2kind, yAxisIndex: ownAxis ? 1 : 0, data: vals, smooth: 0.25, symbol: 'circle', symbolSize: 6,
        lineStyle: {width: 2.4, color: col}, itemStyle: {color: col}});
    });
    var allFmt = function (name, field) { return fmtOf(b, field); };
    opt.tooltip.trigger = 'axis';
    opt.tooltip.axisPointer = {type: kind === 'line' || kind === 'area' ? 'line' : 'shadow', shadowStyle: {color: 'rgba(120,130,150,.08)'}};
    opt.tooltip.formatter = function (ps) {
      var head = ps.length ? ps[0].axisValueLabel || ps[0].name : '';
      return AK.tipRows(head, ps.map(function (p) {
        var ser = opt.series[p.seriesIndex];
        var field = p.seriesIndex >= s.series.length ? y2[p.seriesIndex - s.series.length] : (s.series[p.seriesIndex] || {}).field;
        var v = p.value && p.value.value != null ? p.value.value : p.value;
        return [p.seriesName, AK.fmt(v, allFmt(p.seriesName, field))];
      }));
    };
    var multi = opt.series.length > 1;
    if (multi && b.legend !== false) {
      opt.legend = {top: 0, left: 0, type: 'scroll', icon: 'roundRect', itemWidth: 12, itemHeight: 8, textStyle: {color: T.text2, fontSize: 12}};
      opt.grid.top = 40;
    } else opt.grid.top = (b.y_label ? 30 : 16);
    if (b.x_label) opt.grid.bottom = 24;
    if ((b.ref_lines && b.ref_lines.length) || b.threshold) opt.grid.right = 28;
    if (horizontal && showValues) opt.grid.right = Math.max(opt.grid.right || 16, Math.round(widest * 6.8) + 12);
    if (horizontal && !b.height) {
      // Room for every bar: the chart grows with its categories.
      var per = (b.stack || s.series.length < 2) ? 30 : 12 * s.series.length + 12;
      opt._akHeight = Math.max(chart ? chart.getHeight() : 0, 70 + s.cats.length * per + (multi ? 24 : 0));
    }
    return opt;
  }

  function pie(b, width) {
    var T = theme();
    var rows = AK.data(b.data);
    var yf = ys(b)[0];
    var fmt = AK.fieldFormat(b, yf) || b.format || 'auto';
    var pal = AK.palette();
    var colors = b.colors || {};
    var narrow = width < 520;
    var opt = baseOption(T);
    var data = rows.map(function (r, i) {
      var name = String(r[b.x]);
      var col = AK.resolveColor(colors[name]) || (AK.toneOf(name) && b.tone_names ? AK.toneColor(AK.toneOf(name)) : pal[i % pal.length]);
      return {name: name, value: AK.toNum(r[yf]), itemStyle: {color: col}};
    });
    var total = data.reduce(function (a, d) { return a + (d.value || 0); }, 0);
    opt.tooltip.trigger = 'item';
    opt.tooltip.formatter = function (p) { return AK.tipRows(p.name, [[AK.label(b, yf), AK.fmt(p.value, fmt)], ['%', AK.fmt(p.percent, 'percent:1')]]); };
    opt.legend = {orient: narrow ? 'horizontal' : 'vertical', left: narrow ? 'center' : 'right', top: narrow ? 'bottom' : 'middle', type: 'scroll', icon: 'circle', textStyle: {color: T.text2, fontSize: 12}};
    var donut = b.kind === 'donut';
    opt.series = [{type: 'pie', radius: donut ? ['52%', '76%'] : [0, '74%'], center: narrow ? ['50%', '44%'] : ['40%', '50%'],
      data: data, avoidLabelOverlap: true, itemStyle: {borderColor: T.surface, borderWidth: 2, borderRadius: donut ? 6 : 2},
      label: {show: !narrow, color: T.text2, fontSize: 11, formatter: function (p) { return p.name + '\n' + AK.fmt(p.value, fmt) + ' (' + AK.fmt(p.percent, 'percent:0') + ')'; }},
      labelLine: {length: 8, length2: 8}}];
    if (donut) {
      opt.title = {text: AK.fmt(total, fmt.indexOf('currency') === 0 ? 'currency_compact' : fmt), subtext: b.center_label || AK.t('total'),
        left: narrow ? '50%' : '40%', top: narrow ? '38%' : '44%', textAlign: 'center',
        textStyle: {color: T.text, fontSize: 20, fontWeight: 700}, subtextStyle: {color: T.text3, fontSize: 12}};
    }
    return opt;
  }

  function scatter(b, width) {
    var T = theme();
    var rows = AK.data(b.data);
    var yf = ys(b)[0];
    var pal = AK.palette();
    var groups = b.series ? AK.uniq(rows, b.series) : [null];
    var maxSize = b.size ? Math.max.apply(null, rows.map(function (r) { return AK.toNum(r[b.size]) || 0; })) || 1 : 1;
    var opt = baseOption(T);
    opt.xAxis = {type: 'value', scale: true, splitLine: {lineStyle: {color: T.grid}}, axisLabel: {color: T.text3, formatter: AK.axisFmt(AK.fieldFormat(b, b.x) || b.x_format)}, name: b.x_label || AK.label(b, b.x), nameLocation: 'middle', nameGap: 28, nameTextStyle: {color: T.text3}};
    opt.yAxis = {type: 'value', scale: true, splitLine: {lineStyle: {color: T.grid}}, axisLabel: {color: T.text3, formatter: AK.axisFmt(AK.fieldFormat(b, yf) || b.format)}, name: b.y_label || AK.label(b, yf), nameTextStyle: {color: T.text3}};
    opt.grid.bottom = 26;
    opt.series = groups.map(function (g, i) {
      var rs = g == null ? rows : rows.filter(function (r) { return String(r[b.series]) === String(g); });
      return {name: g == null ? AK.label(b, yf) : String(g), type: 'scatter', itemStyle: {color: pal[i % pal.length], opacity: 0.85},
        symbolSize: function (v, p) { return b.size ? 8 + 26 * Math.sqrt((AK.toNum(rs[p.dataIndex][b.size]) || 0) / maxSize) : 11; },
        data: rs.map(function (r) { return [AK.toNum(r[b.x]), AK.toNum(r[yf])]; }),
        label: b.label ? {show: true, position: 'right', fontSize: 11, color: T.text2, formatter: function (p) { return String(rs[p.dataIndex][b.label]); }} : undefined};
    });
    if (groups.length > 1) opt.legend = {top: 0, left: 0, textStyle: {color: T.text2}};
    opt.tooltip.trigger = 'item';
    opt.tooltip.formatter = function (p) {
      var rs = groups[p.seriesIndex] == null ? rows : rows.filter(function (r) { return String(r[b.series]) === String(groups[p.seriesIndex]); });
      var r = rs[p.dataIndex] || {};
      return AK.tipRows(b.label ? String(r[b.label]) : p.seriesName, [[AK.label(b, b.x), AK.fmt(r[b.x], fmtOf(b, b.x))], [AK.label(b, yf), AK.fmt(r[yf], fmtOf(b, yf))]].concat(b.size ? [[AK.label(b, b.size), AK.fmt(r[b.size], AK.fieldFormat(b, b.size))]] : []));
    };
    return opt;
  }

  function heatmap(b, width) {
    var T = theme();
    var rows = AK.data(b.data);
    var vf = b.value || ys(b)[0];
    var yc = b.y_category || b.series;
    var xs = AK.uniq(rows, b.x), yv = AK.uniq(rows, yc);
    var vals = rows.map(function (r) { return AK.toNum(r[vf]); }).filter(AK.isNum);
    var fmt = AK.fieldFormat(b, vf) || b.format || 'auto';
    var opt = baseOption(T);
    var dateX = isDateField(rows, b.x);
    var hxfmt = b.x_format || (dateX ? AK.dateFormatFor(xs, true) : 'auto');
    opt.xAxis = {type: 'category', data: xs.map(function (x) { return AK.fmt(x, hxfmt); }), splitArea: {show: false}, axisLabel: {color: T.text2, fontSize: 11, hideOverlap: true}, axisLine: {lineStyle: {color: T.border}}};
    opt.yAxis = {type: 'category', data: yv.map(String), axisLabel: {color: T.text2, fontSize: 11}, axisLine: {lineStyle: {color: T.border}}};
    var lo = b.min != null ? b.min : Math.min.apply(null, vals), hi = b.max != null ? b.max : Math.max.apply(null, vals);
    opt.visualMap = {min: lo, max: hi, calculable: false, orient: 'horizontal', left: 'center', bottom: 0, itemWidth: 12, itemHeight: 120, textStyle: {color: T.text3, fontSize: 11},
      formatter: function (v) { return AK.axisFmt(fmt)(v); },
      inRange: {color: b.colors_range || [AK.css('--ak-surface-2'), AK.css('--ak-accent')]}};
    opt.grid.bottom = 44;
    opt.series = [{type: 'heatmap', data: rows.map(function (r) { return [xs.indexOf(r[b.x]), yv.indexOf(r[yc]), AK.toNum(r[vf])]; }),
      label: {show: xs.length * yv.length <= 120, fontSize: 10, color: T.text, formatter: function (p) { return AK.axisFmt(fmt)(p.value[2]); }},
      itemStyle: {borderColor: T.surface, borderWidth: 2, borderRadius: 3}}];
    opt.tooltip.trigger = 'item';
    opt.tooltip.formatter = function (p) { return AK.tipRows(String(yv[p.value[1]]) + ' · ' + opt.xAxis.data[p.value[0]], [[AK.label(b, vf), AK.fmt(p.value[2], fmt)]]); };
    return opt;
  }

  function waterfall(b, width) {
    var T = theme();
    var rows = AK.data(b.data);
    var yf = ys(b)[0];
    var fmt = AK.fieldFormat(b, yf) || b.format || 'auto';
    var cats = rows.map(function (r) { return String(r[b.x]); });
    var base = [], up = [], down = [], run = 0;
    rows.forEach(function (r, i) {
      var v = AK.toNum(r[yf]) || 0;
      var isTotal = b.totals && b.totals.indexOf(String(r[b.x])) >= 0;
      if (isTotal) { base.push(0); up.push(v); down.push(null); run = v; return; }
      if (v >= 0) { base.push(run); up.push(v); down.push(null); run += v; }
      else { run += v; base.push(run); up.push(null); down.push(-v); }
    });
    var opt = baseOption(T);
    opt.xAxis = {type: 'category', data: cats, axisLabel: {color: T.text2, fontSize: 11, hideOverlap: true}, axisLine: {lineStyle: {color: T.border}}, axisTick: {show: false}};
    opt.yAxis = {type: 'value', splitLine: {lineStyle: {color: T.grid}}, axisLabel: {color: T.text3, formatter: AK.axisFmt(fmt)}};
    var lab = {show: true, position: 'top', fontSize: 11, color: T.text2, formatter: function (p) { return p.value == null ? '' : AK.axisFmt(fmt)(p.value); }};
    opt.series = [
      {type: 'bar', stack: 'w', data: base, itemStyle: {color: 'transparent'}, emphasis: {disabled: true}, tooltip: {show: false}},
      {type: 'bar', stack: 'w', name: '+', data: up, itemStyle: {color: AK.css('--ak-accent'), borderRadius: [4, 4, 0, 0]}, label: lab},
      {type: 'bar', stack: 'w', name: '−', data: down, itemStyle: {color: AK.css('--ak-bad'), borderRadius: [4, 4, 0, 0]}, label: lab}
    ];
    opt.tooltip.trigger = 'axis';
    opt.tooltip.formatter = function (ps) { var i = ps[0].dataIndex; return AK.tipRows(cats[i], [[AK.label(b, yf), AK.fmt(rows[i][yf], fmt)]]); };
    return opt;
  }

  function bindSelection(b, chart) {
    var rows = AK.data(b.data);
    var key = b.key || b.x;
    chart.on('click', function (p) {
      var cats = AK.uniq(rows, b.x);
      var cat = cats[p.dataIndex];
      var r = rows.filter(function (x) { return String(x[b.x]) === String(cat); })[0];
      if (r) AK.select(b.link, r[key], chart);
    });
    AK.onSelect(b.link, function (sel) {
      chart.dispatchAction({type: 'downplay'});
      if (sel == null) return;
      var cats = AK.uniq(rows, b.x);
      var i = -1;
      cats.forEach(function (c, j) { var r = rows.filter(function (x) { return String(x[b.x]) === String(c); })[0]; if (r && String(r[key]) === sel) i = j; });
      if (i >= 0) { chart.dispatchAction({type: 'highlight', dataIndex: i}); chart.dispatchAction({type: 'showTip', seriesIndex: 0, dataIndex: i}); }
    });
  }

  AK.blocks.chart = function (b, parent) {
    var kind = b.kind || 'bar';
    var build = {pie: pie, donut: pie, scatter: scatter, heatmap: heatmap, waterfall: waterfall}[kind] || cartesian;
    return mountChart(b, parent, kind, function (w, chart) { return build(b, w, chart); });
  };

  /* Raw ECharts option for chart types the kit doesn't name (radar, treemap, sunburst, funnel, gauge,
     calendar, candlestick, parallel, boxplot ...). Colours and fonts follow the page. */
  AK.blocks.echarts = function (b, parent) {
    var spec = Object.assign({}, b, {show_data: false});
    return mountChart(spec, parent, 'custom', function () {
      var T = theme();
      var opt = JSON.parse(JSON.stringify(b.option || {}));
      var base = baseOption(T);
      if (!opt.color) opt.color = AK.palette();
      if (!opt.textStyle) opt.textStyle = base.textStyle;
      if (opt.tooltip && !opt.tooltip.backgroundColor) opt.tooltip = Object.assign({}, base.tooltip, opt.tooltip);
      return opt;
    });
  };
})();
