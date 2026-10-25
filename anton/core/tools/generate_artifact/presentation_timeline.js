/* Actual date positions, with readable milestone labels at every viewport width. */
(() => {
  const ns = 'http://www.w3.org/2000/svg';
  function svgNode(tag, attributes, text) {
    const node = document.createElementNS(ns, tag);
    for (const [key, value] of Object.entries(attributes || {})) node.setAttribute(key, String(value));
    if (text != null) node.textContent = String(text);
    return node;
  }
  window.CoworkTimeline = (section, source, cleanup) => {
    const items = source.items || [];
    const times = items.map(item => {
      if (typeof item.time !== 'string' || !/^\d{4}-\d{2}-\d{2}(?:T.+)?$/.test(item.time)) return NaN;
      const day = Date.parse(item.time.slice(0, 10));
      if (!Number.isFinite(day) || new Date(day).toISOString().slice(0, 10) !== item.time.slice(0, 10)) return NaN;
      return Date.parse(item.time);
    });
    // Preserve the ordinary milestone list for categorical or unspecified dates.
    if (!times.length || !times.every(Number.isFinite)) return false;
    const view = document.createElement('div'); view.className = 'timeline-axis'; section.append(view);
    function draw() {
      const width = view.clientWidth;
      if (!width) return;
      const left = Math.min(240, width * .42), right = width - 12;
      let minimum = Infinity, maximum = -Infinity;
      for (const time of times) {minimum = Math.min(minimum, time); maximum = Math.max(maximum, time);}
      const x = time => maximum === minimum ? (left + right) / 2
        : left + (time - minimum) / (maximum - minimum) * (right - left);
      const svg = svgNode('svg', {width: '100%', role: 'img', 'aria-label': source.title || ''});
      svg.append(svgNode('desc', {}, items.map(item => [item.time, item.label, item.detail].filter(Boolean).join(' · ')).join('\n')));
      view.replaceChildren(svg);
      function textBlock(value, x, y, available, size, color, weight) {
        const text = svgNode('text', {x, y, fill: color, 'font-size': size, 'font-weight': weight || 400});
        svg.append(text);
        let line = '', lines = 1, span = svgNode('tspan', {x, dy: 0}); text.append(span);
        // Measure actual glyph widths; retain every character, including long identifiers.
        for (const word of String(value || '').split(/\s+/)) {
          const trial = line ? line + ' ' + word : word; span.textContent = trial;
          if (span.getComputedTextLength() <= available) {line = trial; continue;}
          if (line) {span.textContent = line; span = svgNode('tspan', {x, dy: size + 4}); text.append(span); lines++; line = '';}
          for (const character of word) {
            span.textContent = line + character;
            if (line && span.getComputedTextLength() > available) {
              span.textContent = line; span = svgNode('tspan', {x, dy: size + 4}); text.append(span); lines++; line = character;
            } else line += character;
            span.textContent = line;
          }
        }
        span.textContent = line;
        return lines * (size + 4);
      }
      const top = 26, axisBottom = svgNode('line', {x1:left, x2:right, y1:top, y2:top, stroke:'#60778c'});
      svg.append(axisBottom);
      let y = top + 42;
      for (let i = 0; i < items.length; i++) {
        const item = items[i], px = x(times[i]);
        const labelHeight = textBlock(item.label, 0, y, left - 20, 13, '#e6edf3', 500);
        const date = svgNode('text', {y:y - 9, fill:'#c4d3e2', 'font-size':11}, item.time); svg.append(date);
        const dateWidth = date.getComputedTextLength();
        let timeHeight = 0;
        if (dateWidth > right - left) {
          date.remove(); timeHeight = textBlock(item.time, 0, y + labelHeight + 3, left - 20, 11, '#c4d3e2') + 3;
        } else date.setAttribute('x', String(Math.max(left, Math.min(right - dateWidth, px - dateWidth / 2))));
        const detailHeight = item.detail ? textBlock(item.detail, 0, y + labelHeight + timeHeight + 3, left - 20, 11, '#9baabc') + 3 : 0;
        svg.append(svgNode('line', {x1:left, x2:right, y1:y + 4, y2:y + 4, stroke:'#293440'}));
        svg.append(svgNode('circle', {cx:px, cy:y + 4, r:5, fill:'#83b6f5'}));
        y += Math.max(48, labelHeight + timeHeight + detailHeight + 20);
      }
      svg.setAttribute('viewBox', `0 0 ${width} ${y}`); svg.setAttribute('height', String(y));
    }
    draw(); const observer = new ResizeObserver(draw); observer.observe(view);
    cleanup.push(() => observer.disconnect());
    return true;
  };
})();
