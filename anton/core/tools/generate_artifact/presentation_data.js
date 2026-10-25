/* Readable details of the actual ECharts series, including transformed and filtered data. */
(() => {
  const translations = {
    en: ['Data', 'Series', 'Item', 'Value', 'Source', 'Target', 'Previous', 'Next'],
    es: ['Datos', 'Serie', 'Elemento', 'Valor', 'Origen', 'Destino', 'Anterior', 'Siguiente'],
    fr: ['Données', 'Série', 'Élément', 'Valeur', 'Source', 'Cible', 'Précédent', 'Suivant'],
    de: ['Daten', 'Reihe', 'Eintrag', 'Wert', 'Quelle', 'Ziel', 'Zurück', 'Weiter'],
    pt: ['Dados', 'Série', 'Item', 'Valor', 'Origem', 'Destino', 'Anterior', 'Próximo'],
    it: ['Dati', 'Serie', 'Elemento', 'Valore', 'Origine', 'Destinazione', 'Precedente', 'Successivo'],
  };
  const names = ['data', 'series', 'item', 'value', 'source', 'target', 'previous', 'next'];
  function node(tag, text, className) {
    const element = document.createElement(tag);
    if (text != null) element.textContent = String(text);
    if (className) element.className = className;
    return element;
  }
  function readSeries(chart, labels) {
    const groups = [];
    chart.getModel().eachSeries(series => {
      const data = series.getData();
      const label = series.name && !series.name.startsWith('series\u0000')
        ? series.name : labels.series + ' ' + (series.seriesIndex + 1);
      if (['sankey', 'graph'].includes(series.subType)) {
        const edges = series.getEdgeData?.();
        if (!edges || !edges.count()) return;
        const endpoint = value => typeof value === 'number' ? data.getName(value) : value;
        const rows = [];
        for (let i = 0; i < edges.count(); i++) {
          const raw = edges.getRawDataItem(i);
          rows.push([endpoint(raw.source), endpoint(raw.target), edges.get('value', i)]);
        }
        groups.push({label, columns: [labels.source, labels.target, labels.value], rows});
        return;
      }
      const dimensions = data.dimensions.map(name => data.getDimensionInfo(name))
        .filter(info => info && !info.isExtraCoord && !info.isCalculationCoord);
      if (!dimensions.length || !data.count()) return;
      const columns = dimensions.map(info => {
        const axisName = series.coordinateSystem?.getAxis?.(info.coordDim)?.model?.get('name');
        return axisName || info.displayName || (info.name === 'value' ? labels.value : info.name);
      });
      const rows = [];
      for (let i = 0; i < data.count(); i++) {
        rows.push(dimensions.map(info => {
          const value = data.get(info.name, i);
          if (info.ordinalMeta) return info.ordinalMeta.categories[value] ?? value;
          if (info.type === 'time' && Number.isFinite(value)) return new Date(value).toISOString();
          return value;
        }));
      }
      // Named pie/funnel items do not have a category coordinate dimension.
      if (data.getName(0) && !dimensions.some(info => info.ordinalMeta)) {
        columns.unshift(labels.item);
        rows.forEach((row, i) => row.unshift(data.getName(i)));
      }
      groups.push({label, columns, rows});
    });
    return groups;
  }
  window.CoworkChartData = (chart, header, section, config, id, openSections, cleanup) => {
    const language = document.documentElement.lang || 'en';
    const words = translations[language.split('-')[0]] || translations.en;
    const labels = Object.fromEntries(names.map((name, i) => [name, config.dataLabels?.[name] || words[i]]));
    const button = node('button', labels.data, 'data-toggle'); button.type = 'button';
    const panel = node('div', null, 'data-panel'); panel.id = 'values-' + id;
    panel.setAttribute('role', 'region'); panel.setAttribute('aria-label', labels.data);
    button.setAttribute('aria-controls', panel.id);
    const formatter = new Intl.NumberFormat(language, {maximumSignificantDigits: 15});
    const format = value => value == null || typeof value === 'number' && !Number.isFinite(value)
      ? '—' : typeof value === 'number' ? formatter.format(value) : String(value);
    function render(groups) {
      panel.replaceChildren();
      for (const group of groups) {
        const wrap = node('div', null, 'data-group'); wrap.append(node('h3', group.label));
        const table = node('table'), head = node('thead'), tr = node('tr'), body = node('tbody');
        group.columns.forEach(label => {const th = node('th', label); th.scope = 'col'; tr.append(th);});
        head.append(tr); table.append(head, body); wrap.append(table);
        let start = 0;
        const pager = node('div', null, 'data-pager'), range = node('span');
        const previous = node('button', labels.previous), next = node('button', labels.next);
        previous.type = next.type = 'button';
        function page() {
          body.replaceChildren();
          for (const row of group.rows.slice(start, start + 100)) {
            const tr = node('tr'); row.forEach((value, i) => {
              const td = node('td'); td.dataset.label = group.columns[i]; td.append(node('span', format(value))); tr.append(td);
            }); body.append(tr);
          }
          previous.disabled = start === 0; next.disabled = start + 100 >= group.rows.length;
          range.textContent = `${start + 1}–${Math.min(start + 100, group.rows.length)} / ${group.rows.length}`;
        }
        previous.onclick = () => {start = Math.max(0, start - 100); page();};
        next.onclick = () => {start += 100; page();};
        if (group.rows.length > 100) {pager.append(previous, range, next); wrap.append(pager);}
        page(); panel.append(wrap);
      }
    }
    function refresh() {
      let groups;
      try { groups = readSeries(chart, labels); }
      catch (error) { button.hidden = panel.hidden = true; panel.replaceChildren(); console.warn('Chart data details unavailable', error); return; }
      const open = openSections.has(id);
      button.hidden = !groups.length; panel.hidden = !open || !groups.length;
      button.setAttribute('aria-expanded', String(open && !!groups.length));
      // Create numeric details only when the user opens the panel.
      if (open && groups.length) render(groups); else panel.replaceChildren();
    }
    button.onclick = () => {openSections.has(id) ? openSections.delete(id) : openSections.add(id); refresh();};
    header.append(button); section.append(panel); refresh();
    for (const event of ['legendselectchanged', 'datazoom']) chart.on(event, refresh);
    cleanup.unshift(() => {for (const event of ['legendselectchanged', 'datazoom']) chart.off(event, refresh);});
  };
})();
