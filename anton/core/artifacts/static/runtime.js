(() => {
  'use strict';
  const root = document.documentElement;
  const outline = document.querySelector('#outline');
  document.querySelector('[data-action="outline"]').addEventListener('click', event => {
    const open = outline.classList.toggle('open');
    event.currentTarget.setAttribute('aria-expanded', String(open));
  });
  document.querySelector('[data-action="print"]').addEventListener('click', () => window.print());
  const ns = 'http://www.w3.org/2000/svg';
  const svgNode = (tag, attrs = {}, text = '') => {
    const el = document.createElementNS(ns, tag);
    for (const [key, value] of Object.entries(attrs)) el.setAttribute(key, value);
    el.textContent = text;
    return el;
  };
  const colors = [getComputedStyle(root).getPropertyValue('--accent').trim(), '#4787a8', '#a16d39', '#8d6e9d', '#7d8f46', '#bb6e71', '#6f839a', '#a18b64'];
  const numeric = value => value === null ? null : Number(value);
  const compact = value => new Intl.NumberFormat(document.documentElement.lang, {notation: 'compact', maximumFractionDigits: 1}).format(value);
  const mark = (svg, tag, attrs, label) => {
    const el = svgNode(tag, {...attrs, tabindex: '0', role: 'img', 'aria-label': label});
    el.append(svgNode('title', {}, label));
    svg.append(el);
  };
  const draw = (container, spec) => {
    const width = container.clientWidth;
    if (!width) return;
    container.replaceChildren();
    if (!spec.rows.length) {container.textContent = 'No records match this selection.'; return;}
    const series = spec.series;
    const allValues = spec.rows.flatMap(r => series.map(s => numeric(r[s.field]))).filter(v => v !== null);
    if (allValues.some(v => !Number.isFinite(v))) throw new Error('Chart requires finite numeric values');
    if (!allValues.length) {container.textContent = 'No observed numeric values.'; return;}
    const rawLow = Math.min(0, ...allValues), rawHigh = Math.max(0, ...allValues);
    const rough = (rawHigh-rawLow || 1)/4, power=10**Math.floor(Math.log10(rough));
    const tick = [1,2,5,10].find(n=>n*power>=rough)*power;
    const low=Math.floor(rawLow/tick)*tick, high=rawHigh===0 && rawLow===0 ? tick : Math.ceil(rawHigh/tick)*tick;
    const extent=high-low;
    const legend = document.createElement('div'); legend.className = 'legend';
    series.forEach((s, i) => {const item=document.createElement('span'), dot=document.createElement('i'); dot.style.backgroundColor=colors[i]; item.append(dot, document.createTextNode(s.label)); legend.append(item);});
    container.append(legend);
    const bar = spec.kind === 'bar', stackedLabels=bar && width<440;
    const left = bar ? (stackedLabels ? 0 : Math.min(width * .32, 170)) : 52;
    const right = bar ? 68 : 20;
    const top = 18;
    const bottom = 54;
    const labelLines=spec.rows.map(row=>{
      const words=String(row[spec.x] ?? 'Not available').split(/\s+/), lines=[''];
      const chars=Math.max(10,Math.floor((width-16)/6.5));
      for(const word of words){
        let rest=word;
        if(lines[lines.length-1] && lines[lines.length-1].length+word.length+1>chars)lines.push('');
        while(rest.length>chars){lines[lines.length-1]+=rest.slice(0,chars);rest=rest.slice(chars);lines.push('');}
        lines[lines.length-1]+=(lines[lines.length-1]?' ':'')+rest;
      }
      return lines;
    });
    const labelHeight=stackedLabels?Math.max(...labelLines.map(lines=>lines.length))*14:0;
    const height = bar ? spec.rows.length * Math.max(stackedLabels?66:38, series.length * 22 + (stackedLabels?labelHeight+28:14)) + bottom : Math.max(250, Math.min(width * .52, 380));
    const plotW = Math.max(40, width - left - right), plotH = height - top - bottom;
    const svg = svgNode('svg', {viewBox:`0 0 ${width} ${height}`, role:'img', 'aria-label':container.getAttribute('aria-label')});
    const scaleX = value => left + (value - low) / extent * plotW;
    const scaleY = value => top + (high - value) / extent * plotH;
    for(let i=0;i<=4;i++) {
      const value=low+extent*i/4;
      if(bar) {const x=scaleX(value);svg.append(svgNode('line',{x1:x,x2:x,y1:top,y2:height-bottom,class:'grid'}));svg.append(svgNode('text',{x,y:height-bottom+22,'text-anchor':'middle'},compact(value)));}
      else {const y=scaleY(value);svg.append(svgNode('line',{x1:left,x2:width-right,y1:y,y2:y,class:'grid'}));svg.append(svgNode('text',{x:left-10,y:y+4,'text-anchor':'end'},compact(value)));}
    }
    if (bar) {
      const step = plotH / spec.rows.length;
      spec.rows.forEach((row, n) => {
        const label=String(row[spec.x] ?? 'Not available'), y=top+n*step;
        const labelNode=svgNode('text',{x:stackedLabels?0:left-12,y:stackedLabels?y+12:y+step/2+4,'text-anchor':stackedLabels?'start':'end'}, label);
        if(stackedLabels){labelNode.textContent='';labelLines[n].forEach((line,j)=>labelNode.append(svgNode('tspan',{x:0,dy:j?14:0},line)));}
        else if(label.length>22)labelNode.textContent=label.slice(0,20)+'…';
        labelNode.append(svgNode('title',{},label)); svg.append(labelNode);
        series.forEach((s,i)=>{const value=numeric(row[s.field]); if(value===null)return;
          const text=spec.display[n][i], x=scaleX(Math.min(0,value)), end=scaleX(Math.max(0,value));
          const by=y+(stackedLabels?labelHeight+24:step/2)-series.length*10+i*20;
          mark(svg,'rect',{x,y:by,width:Math.max(1,end-x),height:14,fill:colors[i],rx:1},`${label}, ${s.label}: ${text}`);
          if(series.length===1)svg.append(svgNode('text',{x:width-right+8,y:by+12},text));
        });
      });
    } else {
      const scatter = spec.kind === 'scatter';
      const xs = spec.rows.map(row => scatter ? numeric(row[spec.x]) : 0);
      if(scatter && xs.some(x => x === null || !Number.isFinite(x))) throw new Error('Scatter x values must be numeric');
      const minX = scatter ? Math.min(...xs) : 0, maxX = scatter ? Math.max(...xs) : spec.rows.length-1;
      const xAt = index => left + ((scatter ? xs[index] : index) - minX) / (maxX-minX || 1) * plotW;
      series.forEach((s,i)=>{
        let path='';let connected=false;
        spec.rows.forEach((row,n)=>{const value=numeric(row[s.field]); if(value===null){connected=false;return;}
          const x=xAt(n), y=scaleY(value);
          path+=`${connected?'L':'M'}${x},${y}`;connected=true;
          mark(svg,'circle',{cx:x,cy:y,r:scatter?4:3,fill:colors[i]},`${row[spec.x]}, ${s.label}: ${spec.display[n][i]}`);
        });
        if(!scatter)svg.prepend(svgNode('path',{d:path,fill:'none',stroke:colors[i],'stroke-width':2}));
      });
      if(scatter) {
        for(let i=0;i<=4;i++) svg.append(svgNode('text',{x:left+plotW*i/4,y:height-bottom+24,'text-anchor':'middle'},compact(minX+(maxX-minX)*i/4)));
      } else {
        const stride=Math.max(1,Math.ceil(spec.rows.length/Math.max(2,Math.floor(plotW/90))));
        spec.rows.forEach((row,n)=>{if(n%stride===0 || n===spec.rows.length-1)svg.append(svgNode('text',{x:xAt(n),y:height-bottom+24,'text-anchor':n===spec.rows.length-1?'end':n===0?'start':'middle'},String(row[spec.x]).length>16 ? String(row[spec.x]).slice(0,14)+'…' : String(row[spec.x])));});
      }
    }
    svg.append(svgNode('text',{x:left+plotW/2,y:height-5,'text-anchor':'middle',class:'axis-label'},bar ? spec.yLabel : spec.xLabel));
    if(!bar && spec.yLabel)svg.append(svgNode('text',{x:left,y:11,class:'axis-label'},spec.yLabel));
    container.append(svg);
  };
  document.querySelectorAll('.chart').forEach(container => {
    const spec=JSON.parse(container.nextElementSibling.textContent);
    const details=container.parentElement.querySelector('.chart-data');
    let first=true;
    const update=()=>{try {draw(container,spec);if(first && container.clientWidth){details.open=false;first=false;}}catch(error){container.textContent=`Chart unavailable: ${error.message}. The data is shown below.`;details.open=true;console.error(error);}};
    new ResizeObserver(update).observe(container); update();
  });
})();
