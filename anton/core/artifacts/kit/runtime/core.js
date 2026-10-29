/* Artifact Kit runtime: core. Reads the page spec, owns formatting, state, selection,
   tooltips, the detail drawer and library loading. Block modules register renderers on AK.blocks. */
(function () {
  'use strict';
  var AK = window.AK = window.AK || {};
  AK.blocks = AK.blocks || {};
  AK.custom = AK.custom || {};
  var specEl = document.getElementById('ak-spec');
  var SPEC = AK.spec = specEl ? JSON.parse(specEl.textContent) : {blocks: []};
  var LANG = (SPEC.lang || document.documentElement.lang || 'en').toLowerCase().slice(0, 2);
  var LOCALES = {en: 'en-US', es: 'es-ES', fr: 'fr-FR', de: 'de-DE', pt: 'pt-BR', it: 'it-IT', nl: 'nl-NL'};
  var LOCALE = AK.locale = SPEC.locale || LOCALES[LANG] || 'en-US';
  AK.lang = LANG;

  var STRINGS = {
    en: {search: 'Search', all: 'All', rows: '{n} rows', rowsOf: '{n} of {m} rows', none: 'No matching rows', details: 'Details', close: 'Close', source: 'Source', notes: 'Notes', total: 'Total', loading: 'Loading', failed: 'This view could not be drawn', zoomIn: 'Zoom in', zoomOut: 'Zoom out', prev: 'Previous', next: 'Next', slide: 'Slide', data: 'View data', more: 'Show more', select: 'Select an item to see its details', diff: 'Change'},
    es: {search: 'Buscar', all: 'Todos', rows: '{n} filas', rowsOf: '{n} de {m} filas', none: 'No hay filas que coincidan', details: 'Detalles', close: 'Cerrar', source: 'Fuente', notes: 'Notas', total: 'Total', loading: 'Cargando', failed: 'No se pudo dibujar esta vista', zoomIn: 'Acercar', zoomOut: 'Alejar', prev: 'Anterior', next: 'Siguiente', slide: 'Diapositiva', data: 'Ver datos', more: 'Mostrar más', select: 'Selecciona un elemento para ver sus detalles', diff: 'Cambio'},
    fr: {search: 'Rechercher', all: 'Tous', rows: '{n} lignes', rowsOf: '{n} sur {m} lignes', none: 'Aucune ligne ne correspond', details: 'Détails', close: 'Fermer', source: 'Source', notes: 'Notes', total: 'Total', loading: 'Chargement', failed: "Cette vue n'a pas pu être affichée", zoomIn: 'Zoom avant', zoomOut: 'Zoom arrière', prev: 'Précédent', next: 'Suivant', slide: 'Diapositive', data: 'Voir les données', more: 'Afficher plus', select: 'Sélectionnez un élément pour voir ses détails', diff: 'Écart'},
    de: {search: 'Suchen', all: 'Alle', rows: '{n} Zeilen', rowsOf: '{n} von {m} Zeilen', none: 'Keine passenden Zeilen', details: 'Details', close: 'Schließen', source: 'Quelle', notes: 'Hinweise', total: 'Summe', loading: 'Wird geladen', failed: 'Diese Ansicht konnte nicht gezeichnet werden', zoomIn: 'Vergrößern', zoomOut: 'Verkleinern', prev: 'Zurück', next: 'Weiter', slide: 'Folie', data: 'Daten anzeigen', more: 'Mehr anzeigen', select: 'Wählen Sie ein Element, um Details zu sehen', diff: 'Änderung'},
    pt: {search: 'Pesquisar', all: 'Todos', rows: '{n} linhas', rowsOf: '{n} de {m} linhas', none: 'Nenhuma linha corresponde', details: 'Detalhes', close: 'Fechar', source: 'Fonte', notes: 'Notas', total: 'Total', loading: 'Carregando', failed: 'Não foi possível desenhar esta vista', zoomIn: 'Aproximar', zoomOut: 'Afastar', prev: 'Anterior', next: 'Próximo', slide: 'Slide', data: 'Ver dados', more: 'Mostrar mais', select: 'Selecione um item para ver os detalhes', diff: 'Variação'},
    it: {search: 'Cerca', all: 'Tutti', rows: '{n} righe', rowsOf: '{n} di {m} righe', none: 'Nessuna riga corrispondente', details: 'Dettagli', close: 'Chiudi', source: 'Fonte', notes: 'Note', total: 'Totale', loading: 'Caricamento', failed: 'Impossibile disegnare questa vista', zoomIn: 'Ingrandisci', zoomOut: 'Riduci', prev: 'Precedente', next: 'Successivo', slide: 'Diapositiva', data: 'Mostra dati', more: 'Mostra altro', select: 'Seleziona un elemento per vederne i dettagli', diff: 'Variazione'}
  };
  AK.t = function (key, vars) {
    var s = (STRINGS[LANG] || STRINGS.en)[key] || STRINGS.en[key] || key;
    if (vars) Object.keys(vars).forEach(function (k) { s = s.replace('{' + k + '}', vars[k]); });
    return s;
  };

  /* ---------- DOM helpers ---------- */
  AK.esc = function (s) {
    return String(s == null ? '' : s).replace(/[&<>"']/g, function (c) {
      return {'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'}[c];
    });
  };
  AK.el = function (tag, cls, text) {
    var e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text != null) e.textContent = text;
    return e;
  };
  AK.svg = function (tag, attrs) {
    var e = document.createElementNS('http://www.w3.org/2000/svg', tag);
    if (attrs) Object.keys(attrs).forEach(function (k) { if (attrs[k] != null) e.setAttribute(k, attrs[k]); });
    return e;
  };
  AK.css = function (name) {
    return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  };

  /* Inline markdown: **bold**, *italic*, `code`, [text](url). Input is escaped first. */
  function inline(s) {
    s = AK.esc(s);
    s = s.replace(/`([^`]+)`/g, '<code>$1</code>');
    s = s.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
    s = s.replace(/(^|[^*])\*([^*\s][^*]*)\*/g, '$1<em>$2</em>');
    s = s.replace(/\[([^\]]+)\]\((https?:\/\/[^)\s]+)\)/g, '<a href="$2" target="_blank" rel="noopener">$1</a>');
    return s;
  }
  AK.inline = inline;
  AK.md = function (src) {
    var lines = String(src == null ? '' : src).replace(/\r/g, '').split('\n');
    var out = [], para = [], list = null;
    function flushPara() { if (para.length) { out.push('<p>' + inline(para.join(' ')) + '</p>'); para = []; } }
    function flushList() { if (list) { out.push('<' + list.tag + '>' + list.items.map(function (i) { return '<li>' + inline(i) + '</li>'; }).join('') + '</' + list.tag + '>'); list = null; } }
    lines.forEach(function (raw) {
      var line = raw.trim(), m;
      if (!line) { flushPara(); flushList(); return; }
      if ((m = line.match(/^(#{1,3})\s+(.*)$/))) { flushPara(); flushList(); var lv = Math.min(3, m[1].length + 1); out.push('<h' + lv + '>' + inline(m[2]) + '</h' + lv + '>'); return; }
      if ((m = line.match(/^[-*•]\s+(.*)$/))) { flushPara(); if (!list || list.tag !== 'ul') { flushList(); list = {tag: 'ul', items: []}; } list.items.push(m[1]); return; }
      if ((m = line.match(/^\d+[.)]\s+(.*)$/))) { flushPara(); if (!list || list.tag !== 'ol') { flushList(); list = {tag: 'ol', items: []}; } list.items.push(m[1]); return; }
      flushList(); para.push(line);
    });
    flushPara(); flushList();
    return out.join('');
  };

  /* ---------- Formatting ---------- */
  var CURRENCY = SPEC.currency || 'USD';
  var nfCache = {};
  function nf(opts) {
    var key = JSON.stringify(opts);
    if (!nfCache[key]) {
      try { nfCache[key] = new Intl.NumberFormat(LOCALE, opts); } catch (e) { nfCache[key] = new Intl.NumberFormat('en-US', opts); }
    }
    return nfCache[key];
  }
  function currencySymbol() {
    try {
      var parts = nf({style: 'currency', currency: CURRENCY, maximumFractionDigits: 0}).formatToParts(1);
      var p = parts.filter(function (x) { return x.type === 'currency'; })[0];
      return {sym: p ? p.value : CURRENCY, before: parts[0].type === 'currency'};
    } catch (e) { return {sym: CURRENCY, before: true}; }
  }
  var CUR = currencySymbol();
  function isNum(v) { return typeof v === 'number' && isFinite(v); }
  AK.isNum = isNum;
  function toNum(v) {
    if (isNum(v)) return v;
    if (typeof v === 'string' && /^-?\d+(\.\d+)?$/.test(v.trim())) return parseFloat(v);
    return null;
  }
  AK.toNum = toNum;
  function compact(v, digits) {
    var a = Math.abs(v), unit = '', d;
    if (a >= 1e9) { v = v / 1e9; unit = 'B'; } else if (a >= 1e6) { v = v / 1e6; unit = 'M'; } else if (a >= 1e4) { v = v / 1e3; unit = 'K'; }
    if (!unit) return nf({maximumFractionDigits: a < 10 ? 2 : (a < 100 ? 1 : 0)}).format(v);
    var av = Math.abs(v);
    d = digits != null ? digits : (av >= 100 ? 0 : (av >= 10 ? 1 : 2));
    var s = nf({minimumFractionDigits: 0, maximumFractionDigits: d}).format(v);
    var sep = LANG === 'en' ? '' : ' ';
    if (LANG !== 'en' && unit === 'K') unit = 'k';
    if (LANG === 'es' && unit === 'K') unit = 'mil';
    return s + sep + unit;
  }
  var DATE_RE = /^(\d{4})-(\d{2})-(\d{2})(?:[T ](\d{2}):(\d{2}))?/;
  function toDate(v) {
    if (v instanceof Date) return v;
    if (typeof v === 'string') {
      var m = v.match(DATE_RE);
      if (m) return new Date(Date.UTC(+m[1], +m[2] - 1, +m[3], m[4] ? +m[4] : 0, m[5] ? +m[5] : 0));
    }
    if (isNum(v) && v > 1e11) return new Date(v);
    return null;
  }
  AK.toDate = toDate;
  var dfCache = {};
  function df(opts) {
    var key = JSON.stringify(opts);
    if (!dfCache[key]) dfCache[key] = new Intl.DateTimeFormat(LOCALE, Object.assign({timeZone: 'UTC'}, opts));
    return dfCache[key];
  }
  /* fmt(value, format): format names are documented in the kit. "percent" takes percent units (98.4),
     "fraction" takes 0..1. A trailing ":N" sets decimals. */
  AK.fmt = function (v, format) {
    if (v == null || v === '') return '';
    var f = String(format || 'auto'), d = null, m = f.match(/^(.*):(\d+)$/);
    if (m) { f = m[1]; d = +m[2]; }
    if (f === 'text') return String(v);
    if (/^date|^month|^week|^datetime|^time/.test(f)) {
      var dt = toDate(v);
      if (!dt) return String(v);
      if (f === 'date_iso') return dt.toISOString().slice(0, 10);
      if (f === 'date_short') return df({month: 'short', day: 'numeric'}).format(dt);
      if (f === 'month') return df({month: 'short', year: 'numeric'}).format(dt);
      if (f === 'datetime') return df({month: 'short', day: 'numeric', year: 'numeric', hour: '2-digit', minute: '2-digit'}).format(dt);
      if (f === 'weekday') return df({weekday: 'short', month: 'short', day: 'numeric'}).format(dt);
      return df({month: 'short', day: 'numeric', year: 'numeric'}).format(dt);
    }
    var n = toNum(v);
    if (n == null) {
      if (f === 'auto' && typeof v === 'string' && DATE_RE.test(v) && v.length <= 10) return v;
      return String(v);
    }
    switch (f) {
      case 'int': return nf({maximumFractionDigits: 0}).format(n);
      case 'number': return nf({minimumFractionDigits: d != null ? d : 0, maximumFractionDigits: d != null ? d : 2}).format(n);
      case 'percent': return nf({minimumFractionDigits: d != null ? d : 0, maximumFractionDigits: d != null ? d : 1}).format(n) + (LANG === 'fr' || LANG === 'de' ? ' %' : '%');
      case 'fraction': return nf({style: 'percent', minimumFractionDigits: d != null ? d : 0, maximumFractionDigits: d != null ? d : 1}).format(n);
      case 'currency': return nf({style: 'currency', currency: CURRENCY, minimumFractionDigits: d != null ? d : 0, maximumFractionDigits: d != null ? d : (Math.abs(n) < 100 && n % 1 ? 2 : 0)}).format(n);
      case 'currency_compact': {
        var c = compact(n, d);
        if (Math.abs(n) < 1e4) return AK.fmt(n, 'currency');
        return CUR.before ? (n < 0 ? '-' + CUR.sym + c.replace('-', '') : CUR.sym + c) : c + ' ' + CUR.sym;
      }
      case 'compact': return compact(n, d);
      case 'signed': return (n > 0 ? '+' : '') + nf({maximumFractionDigits: d != null ? d : 2}).format(n);
      case 'days': return nf({maximumFractionDigits: d != null ? d : 0}).format(n) + ({es: ' días', fr: ' j', de: ' Tage', pt: ' dias', it: ' giorni'}[LANG] || (Math.abs(n) === 1 ? ' day' : ' days'));
      case 'hours': return nf({maximumFractionDigits: d != null ? d : 0}).format(n) + ' h';
      default: return nf({maximumFractionDigits: d != null ? d : (Number.isInteger(n) ? 0 : 2)}).format(n);
    }
  };
  /* A sensible axis/label format for a field's values, given the field's declared format. */
  AK.axisFmt = function (format) {
    var f = String(format || 'auto').split(':')[0];
    if (f === 'currency' || f === 'currency_compact') return function (v) { return AK.fmt(v, 'currency_compact'); };
    if (f === 'percent' || f === 'fraction') return function (v) { return AK.fmt(v, f + ':0'); };
    if (/^date|^month/.test(f)) return function (v) { return AK.fmt(v, f === 'month' ? 'month' : 'date_short'); };
    return function (v) { return isNum(v) && Math.abs(v) >= 1e4 ? compact(v) : AK.fmt(v, f === 'auto' ? 'number' : format); };
  };

  /* ---------- Tones and palette ---------- */
  var TONE_WORDS = {
    bad: /^(late|delayed|overdue|critical|high|at risk|risk|blocked|failed|error|overload(ed)?|over capacity|short|shortage|breach|retrasad[oa]s?|tard[ií]o|en retard|retard|cr[ií]tic[oa]|alto|haute?|élevée?)$/i,
    warn: /^(medium|warning|watch|pending|moderate|delayed risk|minor|amber|media|moyenne?|en attente|pendiente)$/i,
    good: /^(on time|on-time|ok|delivered|complete(d)?|healthy|low|resolved|green|a tiempo|entregad[oa]|à l'heure|livrée?|bajo|basse?)$/i,
    info: /^(in transit|planned|scheduled|open|en tr[aá]nsito|en transit|prévue?|planificad[oa])$/i
  };
  AK.toneOf = function (value, map) {
    if (value == null) return null;
    if (map && map[value] != null) return map[value];
    var s = String(value).trim();
    for (var t in TONE_WORDS) if (TONE_WORDS[t].test(s)) return t;
    return null;
  };
  AK.palette = function () {
    var dark = document.documentElement.getAttribute('data-ak-theme') === 'dark';
    var base = (SPEC.palette && SPEC.palette.length) ? SPEC.palette : (dark
      ? ['#8b8cff', '#2dd4bf', '#fbbf24', '#f472b6', '#60a5fa', '#a3e635', '#fb923c', '#c084fc', '#94a3b8', '#f87171']
      : ['#4f46e5', '#0d9488', '#d97706', '#db2777', '#2563eb', '#65a30d', '#ea580c', '#7c3aed', '#64748b', '#dc2626']);
    return base.slice();
  };
  AK.toneColor = function (tone) {
    var map = {good: '--ak-good', bad: '--ak-bad', warn: '--ak-warn', info: '--ak-info', accent: '--ak-accent', neutral: '--ak-text-3'};
    return map[tone] ? AK.css(map[tone]) : null;
  };
  /* Stable category -> colour: explicit colours first, then tones for status words, then the palette. */
  AK.colorScale = function (values, explicit) {
    var pal = AK.palette(), map = {}, i = 0;
    values.forEach(function (v) {
      var k = String(v);
      if (map[k]) return;
      if (explicit && explicit[k]) { map[k] = AK.toneColor(explicit[k]) || explicit[k]; return; }
      map[k] = pal[i % pal.length]; i++;
    });
    var f = function (v) { return map[String(v)] || pal[0]; };
    f.domain = function () { return Object.keys(map); };
    return f;
  };
  AK.resolveColor = function (c) { return AK.toneColor(c) || c; };

  /* ---------- Data ---------- */
  AK.data = function (ref) {
    if (ref == null) return [];
    if (Array.isArray(ref)) return ref;
    if (typeof ref === 'string') return (SPEC.data && SPEC.data[ref]) || [];
    return [];
  };
  AK.uniq = function (rows, field) {
    var seen = {}, out = [];
    rows.forEach(function (r) { var v = r[field]; if (v == null) return; var k = String(v); if (!seen[k]) { seen[k] = 1; out.push(v); } });
    return out;
  };
  AK.label = function (block, field) {
    return (block.labels && block.labels[field]) || (SPEC.labels && SPEC.labels[field]) || prettify(field);
  };
  function prettify(field) {
    var s = String(field).replace(/[._]+/g, ' ').replace(/([a-z])([A-Z])/g, '$1 $2').trim();
    return s.charAt(0).toUpperCase() + s.slice(1);
  }
  AK.prettify = prettify;
  AK.fieldFormat = function (block, field) {
    return (block.formats && block.formats[field]) || (SPEC.formats && SPEC.formats[field]) || null;
  };

  /* ---------- State and selection ---------- */
  var state = AK.state = Object.assign({}, SPEC.state || {});
  var subs = {};
  AK.set = function (key, value, source) {
    state[key] = value;
    (subs[key] || []).forEach(function (fn) { try { fn(value, source); } catch (e) { report(e); } });
  };
  AK.on = function (key, fn) { (subs[key] = subs[key] || []).push(fn); };
  /* Selection channels: blocks that share a `link` name highlight the same key. */
  AK.select = function (channel, key, source) { if (channel) AK.set('sel:' + channel, key == null ? null : String(key), source); };
  AK.onSelect = function (channel, fn) { if (channel) AK.on('sel:' + channel, fn); };
  AK.selected = function (channel) { return channel ? state['sel:' + channel] : null; };

  /* ---------- Tooltip ---------- */
  var tip = null;
  AK.tip = function (html, evt) {
    if (!tip) { tip = AK.el('div', 'ak-tip'); tip.setAttribute('role', 'tooltip'); document.body.appendChild(tip); }
    if (html == null) { tip.classList.remove('ak-show'); return; }
    tip.innerHTML = html;
    tip.classList.add('ak-show');
    var x = evt.clientX, y = evt.clientY, w = tip.offsetWidth, h = tip.offsetHeight;
    var left = x + 14, top = y + 14;
    if (left + w > window.innerWidth - 8) left = Math.max(8, x - w - 14);
    if (top + h > window.innerHeight - 8) top = Math.max(8, y - h - 14);
    tip.style.left = left + 'px'; tip.style.top = top + 'px';
  };
  AK.tipRows = function (title, pairs) {
    var out = title ? '<b>' + AK.esc(title) + '</b>' : '';
    pairs.forEach(function (p) {
      if (p[1] == null || p[1] === '') return;
      out += '<div class="ak-tip-row"><span>' + AK.esc(p[0]) + '</span><span>' + AK.esc(p[1]) + '</span></div>';
    });
    return out;
  };
  document.addEventListener('scroll', function () { AK.tip(null); }, true);

  /* ---------- Detail drawer ---------- */
  var drawer = null, back = null;
  AK.detailList = function (pairs) {
    var dl = AK.el('dl', 'ak-dl');
    pairs.forEach(function (p) {
      if (p[1] == null || p[1] === '') return;
      dl.appendChild(AK.el('dt', null, p[0]));
      var dd = AK.el('dd', null, p[1]);
      if (p[2]) { dd.innerHTML = ''; var pill = AK.el('span', 'ak-pill ak-tone-' + p[2], p[1]); dd.appendChild(pill); }
      dl.appendChild(dd);
    });
    return dl;
  };
  AK.openDetail = function (title, pairs, subtitle) {
    if (!drawer) {
      back = AK.el('div', 'ak-drawer-back');
      drawer = AK.el('aside', 'ak-drawer');
      drawer.setAttribute('role', 'dialog');
      drawer.setAttribute('aria-modal', 'false');
      back.addEventListener('click', AK.closeDetail);
      document.addEventListener('keydown', function (e) { if (e.key === 'Escape') AK.closeDetail(); });
      document.body.appendChild(back); document.body.appendChild(drawer);
    }
    drawer.innerHTML = '';
    var close = AK.el('button', 'ak-drawer-close', '×');
    close.setAttribute('aria-label', AK.t('close'));
    close.addEventListener('click', AK.closeDetail);
    drawer.appendChild(close);
    drawer.appendChild(AK.el('div', 'ak-kpi-label', AK.t('details')));
    drawer.appendChild(AK.el('h3', null, title));
    if (subtitle) drawer.appendChild(AK.el('div', 'ak-card-sub', subtitle));
    drawer.appendChild(AK.detailList(pairs));
    drawer.setAttribute('aria-label', title);
    requestAnimationFrame(function () { back.classList.add('ak-show'); drawer.classList.add('ak-show'); });
  };
  AK.closeDetail = function () {
    if (drawer) { drawer.classList.remove('ak-show'); back.classList.remove('ak-show'); }
  };

  /* ---------- Libraries (cdn.jsdelivr.net only; the preview policy allows it) ---------- */
  var LIBS = {
    echarts: 'https://cdn.jsdelivr.net/npm/echarts@5.5.1/dist/echarts.min.js',
    d3: 'https://cdn.jsdelivr.net/npm/d3@7.9.0/dist/d3.min.js',
    topojson: 'https://cdn.jsdelivr.net/npm/topojson-client@3.1.0/dist/topojson-client.min.js'
  };
  var GLOBALS = {echarts: 'echarts', d3: 'd3', topojson: 'topojson'};
  var loading = {};
  AK.lib = function (name) {
    if (window[GLOBALS[name]]) return Promise.resolve(window[GLOBALS[name]]);
    if (!loading[name]) {
      loading[name] = new Promise(function (resolve, reject) {
        var s = document.createElement('script');
        s.src = LIBS[name]; s.async = true; s.crossOrigin = 'anonymous';
        s.onload = function () { resolve(window[GLOBALS[name]]); };
        s.onerror = function () { reject(new Error(name + ' did not load')); };
        document.head.appendChild(s);
      });
    }
    return loading[name];
  };
  var worldCache = {};
  AK.world = function (detail) {
    var res = detail === '50m' ? '50m' : '110m';
    if (!worldCache[res]) {
      worldCache[res] = Promise.all([AK.lib('topojson'), fetch('https://cdn.jsdelivr.net/npm/world-atlas@2.0.2/countries-' + res + '.json').then(function (r) { return r.json(); })])
        .then(function (parts) {
          var topo = parts[1], tj = parts[0];
          return {
            countries: tj.feature(topo, topo.objects.countries),
            land: tj.feature(topo, topo.objects.land),
            borders: tj.mesh(topo, topo.objects.countries, function (a, b) { return a !== b; })
          };
        });
    }
    return worldCache[res];
  };

  /* ---------- Errors: a view that fails says so in place, never in the console ---------- */
  function report(e) { AK.errors = (AK.errors || []).concat([String(e && e.message || e)]); }
  AK.report = report;
  AK.fail = function (box, e) {
    report(e);
    box.innerHTML = '';
    var msg = AK.el('div', 'ak-error', AK.t('failed') + '.');
    box.appendChild(msg);
  };

  /* ---------- Resize: views redraw when their box changes width ---------- */
  AK.watchSize = function (el, fn) {
    var last = 0, timer = null;
    var run = function () {
      var w = Math.round(el.getBoundingClientRect().width);
      if (w && w !== last) { last = w; fn(w); }
    };
    if (window.ResizeObserver) {
      new ResizeObserver(function () { clearTimeout(timer); timer = setTimeout(run, 60); }).observe(el);
    } else {
      window.addEventListener('resize', function () { clearTimeout(timer); timer = setTimeout(run, 80); });
    }
    run();
  };
  AK.viewHeight = function (block, el, fallback) {
    var w = el.getBoundingClientRect().width || 800;
    var h = block.height || fallback || 360;
    if (w < 520) h = Math.round(Math.min(h, Math.max(block.min_height || 240, w * (block.phone_ratio || 0.78))));
    return h;
  };

  /* ---------- Rendering ---------- */
  var counter = 0;
  AK.uid = function (p) { counter += 1; return (p || 'ak') + counter; };
  AK.render = function (block, parent, ctx) {
    if (!block) return null;
    var fn = AK.blocks[block.type];
    var wrap;
    try {
      if (!fn) throw new Error('unknown block ' + block.type);
      wrap = fn(block, parent, ctx || {});
    } catch (e) {
      report(e);
      wrap = AK.el('div', 'ak-error', AK.t('failed') + '.');
      parent.appendChild(wrap);
    }
    if (wrap && block.id && !wrap.id) wrap.id = block.id;
    return wrap;
  };
  AK.renderAll = function (blocks, parent, ctx) {
    (blocks || []).forEach(function (b) { AK.render(b, parent, ctx); });
  };
  /* A titled card around a view. Returns {card, body}. */
  AK.card = function (block, parent, extraClass) {
    var card = AK.el('section', 'ak-card' + (extraClass ? ' ' + extraClass : ''));
    if (block.title || block.subtitle) {
      var head = AK.el('div', 'ak-card-head');
      var hb = AK.el('div');
      if (block.title) hb.appendChild(AK.el('h3', 'ak-card-title', block.title));
      if (block.subtitle) hb.appendChild(AK.el('div', 'ak-card-sub', block.subtitle));
      head.appendChild(hb);
      card.appendChild(head);
      card._head = head;
    }
    var body = AK.el('div', 'ak-card-body');
    card.appendChild(body);
    if (block.note) {
      var foot = AK.el('div', 'ak-card-foot');
      foot.innerHTML = inline(block.note);
      card.appendChild(foot);
    }
    parent.appendChild(card);
    return {card: card, body: body};
  };
  AK.legend = function (items) {
    var lg = AK.el('div', 'ak-legend');
    items.forEach(function (it) {
      var li = AK.el('span', 'ak-legend-item');
      var sw = AK.el('i', it.shape ? 'ak-' + it.shape : '');
      if (it.shape === 'dash') sw.style.color = it.color; else sw.style.background = it.color;
      li.appendChild(sw);
      li.appendChild(document.createTextNode(it.label));
      lg.appendChild(li);
    });
    return lg;
  };

  function header(spec, root) {
    if (!spec.title && !spec.subtitle) return;
    var h = AK.el('header', 'ak-header ak-anim-in');
    var chips = (spec.badges || []).filter(Boolean);
    if (chips.length) {
      var eb = AK.el('div', 'ak-eyebrow');
      chips.forEach(function (c) {
        var label = typeof c === 'string' ? c : c.label;
        var tone = typeof c === 'string' ? null : c.tone;
        eb.appendChild(AK.el('span', 'ak-chip' + (tone ? ' ak-tone-' + tone : ''), label));
      });
      h.appendChild(eb);
    }
    if (spec.title) h.appendChild(AK.el('h1', 'ak-title', spec.title));
    if (spec.subtitle) { var st = AK.el('p', 'ak-subtitle'); st.innerHTML = inline(spec.subtitle); h.appendChild(st); }
    var meta = (spec.meta || []).filter(Boolean);
    if (meta.length) {
      var mt = AK.el('div', 'ak-meta');
      meta.forEach(function (m) { mt.appendChild(AK.el('span', null, m)); });
      h.appendChild(mt);
    }
    root.appendChild(h);
  }
  function footer(spec, root) {
    var notes = (spec.notes || []).filter(Boolean), sources = (spec.sources || []).filter(Boolean);
    if (!notes.length && !sources.length) return;
    var f = AK.el('footer', 'ak-footer');
    sources.forEach(function (s) { var d = AK.el('div'); d.innerHTML = '<strong>' + AK.esc(AK.t('source')) + ':</strong> ' + inline(s); f.appendChild(d); });
    notes.forEach(function (n) { var d = AK.el('div'); d.innerHTML = inline(n); f.appendChild(d); });
    root.appendChild(f);
  }

  function boot() {
    var theme = SPEC.theme || 'light';
    if (theme === 'auto') theme = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
    document.documentElement.setAttribute('data-ak-theme', theme);
    document.documentElement.lang = LANG;
    if (SPEC.accent) {
      document.documentElement.style.setProperty('--ak-accent', SPEC.accent);
      document.documentElement.style.setProperty('--ak-accent-soft', SPEC.accent + '1f');
    }
    if (SPEC.motion !== false) document.documentElement.classList.add('ak-motion');
    var mount = document.getElementById('ak-root') || document.body;
    if (SPEC.mode === 'deck' && AK.deck) { AK.deck(SPEC, mount); return; }
    var page = AK.el('main', 'ak-page');
    var widths = {narrow: '820px', normal: '1240px', wide: '1480px', full: 'none', mobile: '560px'};
    if (SPEC.width && widths[SPEC.width]) page.style.setProperty('--ak-max', widths[SPEC.width]);
    if (SPEC.width === 'mobile') page.classList.add('ak-mobile-first');
    mount.appendChild(page);
    header(SPEC, page);
    var body = AK.el('div', 'ak-stack');
    page.appendChild(body);
    AK.renderAll(SPEC.blocks, body, {top: true});
    var i = 0;
    Array.prototype.forEach.call(body.children, function (c) {
      c.classList.add('ak-anim-in');
      c.style.setProperty('--ak-delay', Math.min(i++ * 60, 420) + 'ms');
    });
    footer(SPEC, page);
  }
  AK.header = header;
  AK.footer = footer;
  AK.boot = boot;
})();
