/* Artifact Kit runtime: tables. Sorting, search, select filters, status pills, inline bars,
   totals, linked selection and a detail view per row. Phones get one card per row. */
(function () {
  'use strict';
  var AK = window.AK;

  function colsOf(b, rows) {
    if (b.columns && b.columns.length) {
      return b.columns.map(function (c) { return typeof c === 'string' ? {field: c} : c; });
    }
    return rows.length ? Object.keys(rows[0]).map(function (k) { return {field: k}; }) : [];
  }

  AK.blocks.table = function (b, parent) {
    var rows = AK.data(b.data).slice();
    var cols = colsOf(b, rows);
    cols.forEach(function (c) {
      c.label = c.label || AK.label(b, c.field);
      c.format = c.format || AK.fieldFormat(b, c.field) || 'auto';
      var sample = rows.map(function (r) { return r[c.field]; }).filter(function (v) { return v != null && v !== ''; });
      // Dates and times in the page's language, times of day kept when the column has them.
      if (c.format === 'auto' && sample.length && sample.every(function (v) { return typeof v === 'string' && AK.toDate(v); }))
        c.format = AK.dateFormatFor(sample, false) || 'auto';
      c.numeric = c.align ? c.align === 'right' : (sample.length > 0 && sample.every(function (v) { return AK.toNum(v) != null; }) && !/^(date|text)/.test(c.format) && !/(^|_)(id|code|year)$/i.test(c.field));
      if (c.pill && !c.tones) c.tones = {};
    });
    var key = b.key || null;
    var link = b.link || null;
    // Rows open a detail view only when asked for (detail="drawer"/"inline"); a linked row click
    // selects (highlights the same key in the linked views) without opening anything.
    var detail = b.detail === undefined || b.detail === null ? null : b.detail;
    if (detail === true) detail = link ? 'inline' : 'drawer';
    var c = AK.card(b, parent);
    var mobile = b.mobile || (cols.length >= 4 ? 'cards' : 'scroll');
    if (mobile === 'cards') c.card.classList.add('ak-cards-mobile');

    // Tools: search, select filters, count
    var tools = AK.el('div', 'ak-table-tools');
    var search = null;
    var showSearch = b.search === true || (b.search !== false && rows.length > 10);
    if (showSearch) {
      search = AK.el('input', 'ak-input');
      search.type = 'search';
      search.placeholder = AK.t('search') + '…';
      search.setAttribute('aria-label', AK.t('search'));
      tools.appendChild(search);
    }
    var filters = (b.filters || []).map(function (f) {
      var spec = typeof f === 'string' ? {field: f} : f;
      var sel = AK.el('select', 'ak-select');
      var label = spec.label || AK.label(b, spec.field);
      sel.setAttribute('aria-label', label);
      var all = AK.el('option', null, label + ': ' + AK.t('all'));
      all.value = '';
      sel.appendChild(all);
      var values = AK.uniq(rows, spec.field).map(String).sort(function (a, z) { return a.localeCompare(z, undefined, {numeric: true}); });
      values.forEach(function (v) { var o = AK.el('option', null, v); o.value = v; sel.appendChild(o); });
      tools.appendChild(sel);
      if (spec.state) AK.on(spec.state, function (v) { sel.value = v == null ? '' : v; apply(); });
      sel.addEventListener('change', function () { if (spec.state) AK.set(spec.state, sel.value || null); apply(); });
      return {field: spec.field, el: sel};
    });
    var count = AK.el('span', 'ak-count');
    if (showSearch || filters.length) { tools.appendChild(count); c.body.appendChild(tools); }

    var wrap = AK.el('div', 'ak-table-wrap');
    if (b.max_height) { wrap.classList.add('ak-scroll-y'); wrap.style.setProperty('--ak-table-max', b.max_height + 'px'); }
    var table = AK.el('table', 'ak-table');
    if (b.caption) { var cap = AK.el('caption', null, b.caption); cap.style.cssText = 'text-align:left;padding:8px 12px;color:var(--ak-text-3);font-size:12.5px;caption-side:bottom'; table.appendChild(cap); }
    var thead = AK.el('thead'), hr = AK.el('tr');
    var sortState = b.sort ? {field: b.sort.field || b.sort, desc: !!b.sort.desc} : null;
    cols.forEach(function (col) {
      var th = AK.el('th', col.numeric ? 'ak-r' : null);
      th.textContent = col.label;
      if (b.sortable !== false) {
        th.classList.add('ak-sortable');
        th.tabIndex = 0;
        var ar = AK.el('span', 'ak-sort', '↕');
        th.appendChild(ar);
        var go = function () {
          if (sortState && sortState.field === col.field) sortState.desc = !sortState.desc;
          else sortState = {field: col.field, desc: col.numeric};
          apply();
        };
        th.addEventListener('click', go);
        th.addEventListener('keydown', function (e) { if (e.key === 'Enter') go(); });
      }
      col.th = th;
      hr.appendChild(th);
    });
    thead.appendChild(hr);
    table.appendChild(thead);
    var tbody = AK.el('tbody');
    table.appendChild(tbody);
    var tfoot = null;
    if (b.totals) { tfoot = AK.el('tfoot'); table.appendChild(tfoot); }
    wrap.appendChild(table);
    c.body.appendChild(wrap);
    var more = null, limit = b.page_size || 200;
    var inlineBox = null;
    if (detail === 'inline') {
      inlineBox = AK.el('div', 'ak-detail-inline');
      inlineBox.appendChild(AK.el('div', 'ak-note', b.detail_hint || AK.t('select')));
      c.body.appendChild(inlineBox);
    }

    var maxima = {};
    cols.forEach(function (col) {
      if (col.bar) maxima[col.field] = Math.max.apply(null, rows.map(function (r) { return Math.abs(AK.toNum(r[col.field]) || 0); }).concat([0])) || 1;
    });

    function cellContent(td, col, row) {
      var v = row[col.field];
      var text = AK.fmt(v, col.format);
      var tone = col.tones ? AK.toneOf(v, col.tones, col.field + ' ' + (col.label || '')) : null;
      if (col.pill || (col.tones && tone)) {
        td.appendChild(AK.el('span', 'ak-pill' + (tone ? ' ak-tone-' + tone : ''), text));
      } else if (col.bar && AK.toNum(v) != null) {
        var cell = AK.el('div', 'ak-bar-cell');
        var bar = AK.el('i');
        bar.style.width = Math.round(Math.abs(AK.toNum(v)) / maxima[col.field] * 64) + 'px';
        if (col.bar !== true) bar.style.background = AK.resolveColor(col.bar);
        cell.appendChild(bar);
        cell.appendChild(AK.el('span', null, text));
        td.appendChild(cell);
      } else {
        td.textContent = text;
      }
    }
    function rowTone(row) {
      if (!b.row_tone) return null;
      var spec = b.row_tone;
      var v = row[spec.field];
      return AK.toneOf(v, spec.map || spec.tones || null, spec.field);
    }
    function detailPairs(row) {
      // Coordinates are for maps, not for people: left out unless asked for.
      var fields = b.detail_fields || Object.keys(row).filter(function (f) { return !/(^|_|[a-z])(lat|lon|lng|latitude|longitude)$/i.test(f); });
      return fields.map(function (f) {
        var col = cols.filter(function (cc) { return cc.field === f; })[0];
        var fmt = (col && col.format) || AK.fieldFormat(b, f) || 'auto';
        var tone = col && col.tones ? AK.toneOf(row[f], col.tones, f + ' ' + (col.label || '')) : null;
        return [(col && col.label) || AK.label(b, f), AK.fmt(row[f], fmt), tone];
      });
    }
    function titleOf(row) {
      var f = b.detail_title || key || cols[0].field;
      return String(row[f] == null ? '' : row[f]);
    }
    var rowEls = [];
    function showInline(row) {
      if (!inlineBox) return;
      inlineBox.innerHTML = '';
      if (!row) { inlineBox.appendChild(AK.el('div', 'ak-note', b.detail_hint || AK.t('select'))); return; }
      var h = AK.el('div', 'ak-card-title', titleOf(row));
      inlineBox.appendChild(h);
      inlineBox.appendChild(AK.detailList(detailPairs(row)));
    }
    function onRow(row) {
      if (link && key) AK.select(link, row[key], table);
      if (detail === 'drawer') AK.openDetail(titleOf(row), detailPairs(row), b.detail_subtitle || null);
      if (detail === 'inline' && !(link && key)) showInline(row);
    }

    function apply() {
      var q = search ? search.value.trim().toLowerCase() : '';
      var view = rows.filter(function (r) {
        for (var i = 0; i < filters.length; i++) {
          var fv = filters[i].el.value;
          if (fv && String(r[filters[i].field]) !== fv) return false;
        }
        if (!q) return true;
        return cols.some(function (col) { return String(AK.fmt(r[col.field], col.format)).toLowerCase().indexOf(q) >= 0 || String(r[col.field]).toLowerCase().indexOf(q) >= 0; });
      });
      if (sortState) {
        var f = sortState.field, dir = sortState.desc ? -1 : 1;
        view.sort(function (a, z) {
          var x = a[f], y = z[f], nx = AK.toNum(x), ny = AK.toNum(y);
          if (x == null) return 1; if (y == null) return -1;
          if (nx != null && ny != null) return (nx - ny) * dir;
          return String(x).localeCompare(String(y), undefined, {numeric: true}) * dir;
        });
        cols.forEach(function (col) {
          if (col.field === f) col.th.setAttribute('aria-sort', sortState.desc ? 'descending' : 'ascending'); else col.th.removeAttribute('aria-sort');
          var arrow = col.th.querySelector('.ak-sort');
          if (arrow) arrow.textContent = col.field === f ? (sortState.desc ? '↓' : '↑') : '↕';
        });
      }
      tbody.innerHTML = '';
      rowEls = [];
      var shown = view.slice(0, limit);
      shown.forEach(function (r) {
        var tr = AK.el('tr');
        var tone = rowTone(r);
        if (tone === 'bad' || tone === 'warn') tr.classList.add('ak-row-' + tone);
        cols.forEach(function (col) {
          var td = AK.el('td', (col.numeric ? 'ak-r' : '') + (col.wrap ? ' ak-wrap' : ''));
          td.setAttribute('data-label', col.label);
          cellContent(td, col, r);
          tr.appendChild(td);
        });
        if (detail || (link && key)) {
          tr.classList.add('ak-clickable');
          tr.tabIndex = 0;
          tr.addEventListener('click', function () { onRow(r); });
          tr.addEventListener('keydown', function (e) { if (e.key === 'Enter') onRow(r); });
        }
        if (key) tr.setAttribute('data-key', String(r[key]));
        tbody.appendChild(tr);
        rowEls.push({tr: tr, row: r});
      });
      if (!view.length) {
        var tr = AK.el('tr'), td = AK.el('td', 'ak-empty', b.empty_text || AK.t('none'));
        td.colSpan = cols.length; tr.appendChild(td); tbody.appendChild(tr);
      }
      if (more) { more.remove(); more = null; }
      if (view.length > shown.length) {
        more = AK.el('button', 'ak-tab', AK.t('more') + ' (' + (view.length - shown.length) + ')');
        more.type = 'button';
        more.style.marginTop = '8px';
        more.addEventListener('click', function () { limit += 200; apply(); });
        c.body.insertBefore(more, inlineBox);
      }
      count.textContent = view.length === rows.length ? AK.t('rows', {n: rows.length}) : AK.t('rowsOf', {n: view.length, m: rows.length});
      if (tfoot) {
        tfoot.innerHTML = '';
        var fr = AK.el('tr');
        cols.forEach(function (col, i) {
          var td = AK.el('td', col.numeric ? 'ak-r' : null);
          td.setAttribute('data-label', col.label);
          var how = b.totals[col.field];
          if (how) {
            var vals = view.map(function (r) { return AK.toNum(r[col.field]); }).filter(AK.isNum);
            var s = vals.reduce(function (a, x) { return a + x; }, 0);
            var v = how === 'mean' || how === 'avg' ? (vals.length ? s / vals.length : null) : how === 'count' ? vals.length : s;
            td.textContent = AK.fmt(v, how === 'count' ? 'int' : col.format);
          } else if (i === 0) td.textContent = b.totals_label || AK.t('total');
          if (!td.textContent) td.className += ' ak-empty-cell';
          fr.appendChild(td);
        });
        tfoot.appendChild(fr);
      }
      mark(AK.selected(link));
    }
    function mark(sel) {
      rowEls.forEach(function (x) { x.tr.classList.toggle('ak-selected', sel != null && key && String(x.row[key]) === sel); });
      if (inlineBox && link && key) {
        var hit = sel == null ? null : rows.filter(function (r) { return String(r[key]) === sel; })[0];
        showInline(hit || null);
      }
    }
    if (search) search.addEventListener('input', apply);
    if (link) AK.onSelect(link, function (sel, source) {
      mark(sel);
      if (sel != null && source !== table) {
        var hit = rowEls.filter(function (x) { return String(x.row[key]) === sel; })[0];
        if (hit && wrap.classList.contains('ak-scroll-y')) hit.tr.scrollIntoView({block: 'nearest'});
      }
    });
    apply();
    return c.card;
  };
})();
