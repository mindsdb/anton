/* Reconcile shown figures with the bound data. Findings are console errors so the browser check returns them for repair. */
(() => {
  const draw = window.CoworkPage, format = window.CoworkNumber, formatted = new Map();
  window.CoworkNumber = (value, ...rest) => {
    const text = format(value, ...rest);
    if (typeof value === 'number' && formatted.size < 20000) formatted.set(text, value);
    return text;
  };
  // A formatted string read back as the number it shows.
  function amount(value) {
    if (typeof value === 'number') return value;
    if (typeof value !== 'string') return undefined;
    if (formatted.has(value)) return formatted.get(value);
    const plain = value.trim();
    return /^-?\d{1,3}(,\d{3})*(\.\d+)?$|^-?\d+(\.\d+)?$/.test(plain) ? Number(plain.replace(/,/g, '')) : undefined;
  }
  const blank = value => typeof value === 'number' && (value === 0 || !Number.isFinite(value)) || value === '' || amount(value) === 0;
  const english = new Intl.NumberFormat('en', {maximumFractionDigits: 2});
  const quote = text => JSON.stringify(String(text ?? '').slice(0, 80));
  const join = (text, reason) => [text, reason].filter(Boolean).join(' · ');
  const list = items => items.slice(0, 4).join(', ') + (items.length > 4 ? ' and ' + (items.length - 4) + ' more' : '');
  function hasRows(value, depth = 0) {
    if (Array.isArray(value)) return value.length > 0;
    return depth < 7 && !!value && typeof value === 'object' && Object.values(value).some(item => hasRows(item, depth + 1));
  }
  function facts() {
    let found = [];
    try { found = JSON.parse(document.getElementById('cowork-figure-checks')?.textContent || '{}').doubleCounts || []; }
    catch {}
    return new Map(found.map(fact => [Math.round(fact.value), fact]));
  }
  function chartValues(option) {
    const values = [], series = Array.isArray(option?.series) ? option.series : option?.series ? [option.series] : [];
    let items = 0;
    for (const s of series) {
      const entries = (s.links || s.edges)?.length ? (s.links || s.edges) : s.data || [];
      items += entries.length;
      for (const item of entries) {
        const value = item && typeof item === 'object' && !Array.isArray(item) ? item.value : item;
        const last = Array.isArray(value) ? value[value.length - 1] : value;
        if (typeof last === 'number') values.push(last);
      }
    }
    return {items, values};
  }
  const ZERO = ' while the bound data has rows. Check the filter, join, key or date match behind it against the '
    + 'observed values in the data contract. If this is the true answer, add zeroReason (why, in the page language) ';

  // Shows a true zero's reason and records unexplained zeros and double counts.
  function review(page, where, findings, doubled, sourced) {
    if (!page || typeof page !== 'object') return page;
    const shown = [], result = {...page};
    const zero = (what, owner) => sourced && findings.push(what + where + ZERO + 'to the ' + owner + '; the page shows it.');
    const lead = page.lead;
    if (lead && lead.value !== undefined) {
      if (!blank(lead.value)) shown.push([lead.value, 'headline ' + quote(lead.text)]);
      else if (lead.zeroReason) result.lead = {...lead, detail: join(lead.detail, lead.zeroReason)};
      else zero('headline ' + quote(lead.text) + ' shows ' + (lead.value === '' ? 'nothing' : String(lead.value)), 'lead');
    }
    result.metrics = (page.metrics || []).map(metric => {
      if (!blank(metric?.value)) { shown.push([metric?.value, 'metric ' + quote(metric?.label)]); return metric; }
      if (metric.zeroReason) return {...metric, detail: join(metric.detail, metric.zeroReason)};
      zero('metric ' + quote(metric.label) + ' shows ' + (metric.value === '' ? 'nothing' : String(metric.value)), 'metric');
      return metric;
    });
    if (!page.metrics) delete result.metrics;
    result.sections = (page.sections || []).map(section => {
      if (!section || typeof section !== 'object') return section;
      const name = section.type + ' ' + quote(section.title), reasons = [];
      if (section.type === 'table') {
        const rows = section.rows || [];
        if (!rows.length) section.zeroReason ? reasons.push(section.zeroReason) : zero(name + ' has no rows', 'section');
        const zeros = [], blocks = new Map();
        for (const column of rows.length ? section.columns || [] : []) {
          const values = rows.map(row => amount(row?.[column.key])).filter(value => value !== undefined);
          values.forEach(value => shown.push([value, name + ', column ' + quote(column.label)]));
          if (!values.length) continue;
          blocks.set(column.unit ?? '', [...blocks.get(column.unit ?? '') || [], column]);
          if (values.every(value => value === 0 || !Number.isFinite(value))) {
            column.zeroReason ? reasons.push(column.zeroReason) : zeros.push(quote(column.label));
          }
        }
        if (zeros.length) zero(name + ' shows 0 in every row of column' + (zeros.length > 1 ? 's ' : ' ') + list(zeros), 'column');
        // A row of zeros across a block of same-unit columns, such as weeks, while other rows have values.
        for (const [unit, block] of blocks) {
          if (block.length < 3) continue;
          const empty = rows.filter(row => block.every(column => amount(row?.[column.key]) === 0));
          if (!empty.length || empty.length === rows.length) continue;
          const names = empty.map(row => quote(Object.values(row).find(value => typeof value === 'string' && amount(value) === undefined) ?? '?'));
          section.zeroReason ? reasons.push(section.zeroReason)
            : zero(name + ' shows 0 in all ' + block.length + (unit ? ' ' + quote(unit) : '') + ' columns for row'
              + (names.length > 1 ? 's ' : ' ') + list(names), 'section');
        }
      } else if (section.type === 'chart' && !section.option?.dataset) {
        const {items, values} = chartValues(section.option);
        values.forEach(value => shown.push([value, name]));
        // Items without numbers, such as a graph's unweighted links, are not judged.
        if (!items || values.length === items && values.every(value => value === 0 || !Number.isFinite(value))) {
          section.zeroReason ? reasons.push(section.zeroReason) : zero(name + (items ? ' plots only 0' : ' plots no values'), 'section');
        }
      } else if (['map', 'globe'].includes(section.type) && !(section.points?.length || section.routes?.length)) {
        section.zeroReason ? reasons.push(section.zeroReason) : zero(name + ' shows no points or routes', 'section');
      } else if (section.type === 'timeline' && !section.items?.length) {
        section.zeroReason ? reasons.push(section.zeroReason) : zero(name + ' has no items', 'section');
      }
      return reasons.length ? {...section, caption: join(section.caption, [...new Set(reasons)].join(' · '))} : section;
    });
    if (!page.sections) delete result.sections;
    for (const [value, label] of shown) {
      const number = amount(value);
      if (typeof number !== 'number' || !Number.isFinite(number)) continue;
      const fact = doubled.get(Math.round(number));
      if (fact && Math.abs(fact.value - number) <= 1e-6 * Math.max(1, Math.abs(fact.value))) findings.push({fact, place: label + where});
    }
    return result;
  }

  const twice = (fact, places) => english.format(fact.value) + ' (shown in ' + places.slice(0, 3).join('; ')
    + (places.length > 3 ? '; and ' + (places.length - 3) + ' more places' : '') + ') counts rows twice: it sums '
    + fact.measure + ' over every element of ' + fact.path + (fact.scope ? ' where ' + fact.scope : '')
    + ', including elements where ' + fact.field + ' = ' + JSON.stringify(fact.marker)
    + ', which hold the sum of the other elements. Counted once it is ' + english.format(fact.parts)
    + '. Sum the other elements, or use only the total elements.';

  // Each control option, the others at their defaults.
  function states(config) {
    const defaults = Object.fromEntries((config.controls || []).map(c => [c.id, c.value ?? c.options?.[0]?.value ?? '']));
    const found = [{selection: defaults, where: ''}];
    for (const control of config.controls || []) for (const option of control.options || []) {
      if (option.value !== defaults[control.id] && found.length < 24) {
        found.push({selection: {...defaults, [control.id]: option.value}, where: ' (' + control.label + ': ' + option.label + ')'});
      }
    }
    return found;
  }

  window.CoworkPage = async config => {
    if (!config || typeof config !== 'object') return draw(config);
    const findings = [], doubled = facts(), sourced = hasRows(window.ARTIFACT_DATA);
    const shown = typeof config.render === 'function'
      ? {...config, render: selection => review(config.render(selection), '', [], doubled, sourced)}
      : review(config, '', findings, doubled, sourced);
    if (config.overview) shown.overview = review(config.overview, '', findings, doubled, sourced);
    await draw(shown);
    if (typeof config.render === 'function') {
      for (const {selection, where} of states(config)) {
        // A state that throws is reported by the page itself when it is opened.
        try { review(config.render(CoworkSelection(selection)), where, findings, doubled, sourced); } catch {}
      }
    }
    // One message per wrong figure, naming where it is shown.
    const places = new Map(), messages = [];
    for (const item of findings) {
      if (typeof item === 'string') messages.push(item);
      else if (places.has(item.fact)) places.get(item.fact).add(item.place);
      else { places.set(item.fact, new Set([item.place])); messages.push(item.fact); }
    }
    const distinct = [...new Set(messages.map(item => typeof item === 'string' ? item : twice(item, [...places.get(item)])))];
    distinct.slice(0, 8).forEach(message => console.error('Figure check: ' + message));
    if (distinct.length > 8) console.error('Figure check: ' + (distinct.length - 8) + ' more figures have the same problems.');
  };
})();
