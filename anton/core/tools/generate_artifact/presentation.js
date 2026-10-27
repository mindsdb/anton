/* Self-contained presentation primitives. Data, scope and conclusions come from the caller. */
(() => {
  const library = new Map();
  const urls = {
    echarts: 'https://cdn.jsdelivr.net/npm/echarts@5.5.1/dist/echarts.min.js',
    d3: 'https://cdn.jsdelivr.net/npm/d3@7.9.0/dist/d3.min.js',
    topojson: 'https://cdn.jsdelivr.net/npm/topojson-client@3.1.0/dist/topojson-client.min.js',
    globe: 'https://cdn.jsdelivr.net/npm/globe.gl@2.34.2/dist/globe.gl.min.js',
  };
  const atlas = 'https://cdn.jsdelivr.net/npm/world-atlas@2.0.2/countries-110m.json';
  const normalized = value => String(value).normalize('NFKD').toLowerCase().replace(/[^\p{L}\p{N}]/gu,'');
  function load(name) {
    if (!library.has(name)) library.set(name, new Promise((resolve, reject) => {
      const script = document.createElement('script');
      script.src = urls[name]; script.onload = resolve;
      script.onerror = () => reject(new Error('Could not load ' + name));
      document.head.append(script);
    }));
    return library.get(name);
  }
  function el(tag, text, className) {
    const node = document.createElement(tag);
    text = CoworkText(text);
    if (text !== undefined && text !== null) node.textContent = String(text);
    if (className) node.className = className;
    return node;
  }
  function tone(node, value) {
    if (['warning', 'danger', 'good'].includes(value)) node.classList.add('tone-' + value);
  }
  window.CoworkNumber = (value, unit = '', lang = document.documentElement.lang || 'en') => {
    if (value === null || value === undefined || typeof value === 'number' && !Number.isFinite(value)) return '—';
    if (typeof value !== 'number') {CoworkText(value);return String(value);}
    if (/^[A-Z]{3}$/.test(unit)) return new Intl.NumberFormat(lang, {style: 'currency', currency: unit, maximumFractionDigits: 2}).format(value);
    return new Intl.NumberFormat(lang, {maximumFractionDigits: 2}).format(value) + (unit ? (unit === '%' ? '' : ' ') + unit : '');
  };
  const escape = text => String(text ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  let countries;
  async function geography() {
    await Promise.all([load('d3'), load('topojson')]);
    if (!countries) countries = d3.json(atlas).then(world => topojson.feature(world, world.objects.countries));
    return countries;
  }
  function coordinate(pair) {
    if (!Array.isArray(pair) || pair.length !== 2 || !pair.every(v => typeof v === 'number' && Number.isFinite(v)) || Math.abs(pair[0]) > 180 || Math.abs(pair[1]) > 90) throw new Error('A geographic coordinate must be a real [longitude, latitude] pair.');
    return pair;
  }
  async function drawGeography(node, section, cleanup) {
    const land = await geography();
    const points = section.points || [], routes = section.routes || [];
    for (const p of points) coordinate([p.lon, p.lat]);
    for (const r of routes) { coordinate(r.from); coordinate(r.to); }
    if (!node.isConnected) return;
    if (section.type === 'globe') {
      await load('globe'); if (!node.isConnected) return;
      node.replaceChildren();
      const globe = Globe({animateIn: false})(node).width(node.clientWidth).height(node.clientHeight)
        .backgroundColor('#101923').showAtmosphere(false).globeImageUrl(null).bumpImageUrl(null)
        .polygonsData(land.features).polygonCapColor(() => '#29445a').polygonSideColor(() => '#182b3b')
        .polygonStrokeColor(() => '#6d8b9f').polygonAltitude(.002)
        .pointsData(points).pointLat(p => p.lat).pointLng(p => p.lon).pointRadius(.35).pointAltitude(.02)
        .pointColor(p => p.color || '#f0bc62').pointLabel(p => escape(p.label || p.name))
        .labelsData(points.filter(p=>p.label)).labelLat(p=>p.lat).labelLng(p=>p.lon)
        .labelText(p=>p.label).labelColor(()=>'#e6edf3').labelSize(.55).labelAltitude(.035)
        .arcsData(routes).arcStartLat(r => r.from[1]).arcStartLng(r => r.from[0])
        .arcEndLat(r => r.to[1]).arcEndLng(r => r.to[0]).arcColor(r => r.color || '#83b6f5')
        .arcStroke(.45).arcLabel(r => escape(r.label || ''));
      const center=section.center;
      if (center) { coordinate(center); globe.pointOfView({lng: center[0], lat: center[1], altitude: 2.2}, 0); }
      const observer = new ResizeObserver(() => globe.width(node.clientWidth).height(node.clientHeight));
      observer.observe(node); cleanup.push(() => {observer.disconnect(); globe._destructor();});
    } else if (section.library === 'echarts') {
      await load('echarts'); if (!node.isConnected) return;
      echarts.registerMap('cowork-world', land);
      await drawChart(node, {option: {geo: {map: 'cowork-world', roam: true, itemStyle: {areaColor:'#233b50',borderColor:'#728a9b'}, label:{show:false}},
        tooltip: {formatter: p => escape(p.data?.displayLabel || (typeof p.data?.label==='string'?p.data.label:p.name) || '')}, series: [
          {type:'scatter',coordinateSystem:'geo',data:points.map(p=>({name:p.name,displayLabel:p.label,value:[p.lon,p.lat],label:{show:!!p.label,formatter:()=>p.label,position:'right',color:'#e6edf3'}})),symbolSize:9,itemStyle:{color:'#f0bc62'}},
          {type:'lines',coordinateSystem:'geo',data:routes.map(r=>({coords:[r.from,r.to],label:r.label})),lineStyle:{color:'#83b6f5',width:2,opacity:.85}}
        ]}}, cleanup);
    } else {
      node.replaceChildren();
      const w = node.clientWidth, h = node.clientHeight;
      const projection = d3.geoNaturalEarth1().fitExtent([[12,12],[w-12,h-12]], land);
      const path = d3.geoPath(projection), svg = d3.select(node).append('svg').attr('viewBox', `0 0 ${w} ${h}`).attr('aria-label', section.title || '');
      const group = svg.append('g');
      group.selectAll('path.land').data(land.features).join('path').attr('class','land').attr('d',path).attr('fill','#233b50').attr('stroke','#647f93').attr('stroke-width',.55);
      group.selectAll('path.route').data(routes).join('path').attr('class','route').attr('d',r=>path({type:'LineString',coordinates:[r.from,r.to]})).attr('fill','none').attr('stroke',r=>r.color||'#83b6f5').attr('stroke-width',2).append('title').text(r=>r.label||'');
      group.selectAll('circle').data(points).join('circle').attr('cx',p=>projection([p.lon,p.lat])[0]).attr('cy',p=>projection([p.lon,p.lat])[1]).attr('r',4).attr('fill',p=>p.color||'#f0bc62').append('title').text(p=>p.label||p.name||'');
      const placed=[];
      for(const point of points.filter(p=>p.label)){
        const [px,py]=projection([point.lon,point.lat]);
        const text=group.append('text').text(point.label).attr('font-size',11).attr('fill','#e6edf3').attr('stroke','#101923').attr('stroke-width',3).attr('paint-order','stroke');
        const width=text.node().getComputedTextLength(),x=Math.max(8,Math.min(w-width-8,px+8));
        let y=Math.max(14,Math.min(h-8,py-8));
        for(let n=0;n<points.length&&placed.some(b=>x<b.x+b.w+5&&x+width>b.x-5&&Math.abs(y-b.y)<16);n++)y=Math.min(h-8,y+17);
        group.insert('line',()=>text.node()).attr('x1',px).attr('y1',py).attr('x2',x).attr('y2',y-4).attr('stroke','#8ba7bc').attr('stroke-width',.7);
        text.attr('x',x).attr('y',y);placed.push({x,y,w:width});
      }
      svg.call(d3.zoom().filter(event=>event.type!=='wheel'||event.ctrlKey).scaleExtent([1,8]).on('zoom',event=>group.attr('transform',event.transform)));
    }
  }
  async function drawChart(node, section, cleanup) {
    await load('echarts'); if (!node.isConnected) return;
    const given = section.option || {};
    const seriesList=Array.isArray(given.series)?given.series:given.series?[given.series]:[];
    const geos=given.geo?(Array.isArray(given.geo)?given.geo:[given.geo]):[];
    const geographicMaps=[...new Set([...geos.map(g=>g.map),...seriesList.filter(s=>s.type==='map').map(s=>s.map)].filter(Boolean))];
    for(const name of geographicMaps){
      if(echarts.getMap(name))continue;
      if(!['world','world-map','cowork-world'].includes(name))throw new Error('Unregistered map geometry '+name+'. Use the map component or provide actual supported geography.');
      echarts.registerMap(name,await geography());
    }
    if(!node.isConnected)return;
    node.replaceChildren();
    const chart = echarts.init(node, 'dark');
    const observer = new ResizeObserver(() => chart.resize()); observer.observe(node);
    cleanup.push(() => {observer.disconnect();chart.dispose();});
    const base = {backgroundColor:'transparent',color:['#83b6f5','#efbd67','#86cdb5','#cf9ada','#ed8b8b'],
      animationDuration:250,textStyle:{fontFamily:'system-ui',color:'#d4dfe9'},
      grid:{left:14,right:22,top:48,bottom:58,containLabel:true},
      tooltip:{trigger:'axis',confine:true},legend:{type:'scroll',bottom:0,textStyle:{color:'#b5c4d2'}}};
    const compact=node.clientWidth<600;
    const axis = (a, vertical=false) => a ? {...a, nameLocation:vertical?'end':'middle',nameGap:vertical?10:37,
      ...(vertical?{nameRotate:0,nameTextStyle:{...a.nameTextStyle,align:'left',width:Math.max(120,node.clientWidth*.8),overflow:'break',lineHeight:14}}:{}),
      ...(compact&&a.type==='value'?{splitNumber:3}:{}),
      axisLabel:{hideOverlap:true,...a.axisLabel,...(vertical&&a.type==='category'&&compact
        ?{width:Math.max(70,Math.min(110,node.clientWidth*.27)),overflow:'break'}:{})}} : a;
    const option = {...base,...given,grid:{...base.grid,...given.grid}};
    if(compact){option.grid={...base.grid,left:8,right:18};option.legend={...base.legend,...given.legend,top:null,bottom:0,orient:'horizontal',type:'scroll'};}
    if (given.xAxis) option.xAxis = Array.isArray(given.xAxis) ? given.xAxis.map(axis) : axis(given.xAxis);
    if (given.yAxis) option.yAxis = Array.isArray(given.yAxis) ? given.yAxis.map(a=>axis(a,true)) : axis(given.yAxis,true);
    chart.setOption(option);
    if(geographicMaps.length){
      for(const series of seriesList)if(series.coordinateSystem==='geo')for(const item of series.data||[]){
        const pairs=series.type==='lines'?item.coords:[Array.isArray(item)?item:item.value];
        for(const pair of pairs||[])if(Array.isArray(pair))coordinate(pair.slice(0,2));
      }
    }
    return chart;
  }
  function summary(parent, page, prefix) {
    if (page.lead) {
      const lead=el('section',null,'lead');lead.id=prefix+'-finding';tone(lead,page.lead.tone);lead.append(el('h2',page.lead.text));
      if(page.lead.value!==undefined)lead.append(el('strong',page.lead.value,'lead-value'));
      if(page.lead.detail)lead.append(el('p',page.lead.detail));parent.append(lead);
    }
    if(page.metrics?.length){const strip=el('section',null,'metrics');strip.id=prefix+'-metrics';
      for(const m of page.metrics){const cell=el('div',null,'metric');cell.append(el('div',m.label,'metric-label'));const value=el('div',CoworkNumber(m.value,m.unit),'metric-value');tone(value,m.tone);cell.append(value);if(m.detail)cell.append(el('div',m.detail,'metric-detail'));strip.append(cell);}parent.append(strip);}
  }
  window.CoworkPage = async config => {
    if (!config || typeof config.title !== 'string') throw new Error('The page needs a title.');
    document.documentElement.lang = config.lang || document.documentElement.lang || 'en';
    document.title = config.title;
    const root = document.getElementById('cowork-page'); root.replaceChildren();
    const header = el('header'); header.id = 'page-header'; header.append(el('h1',config.title));
    if (config.subtitle) header.append(el('p',config.subtitle,'subtitle')); root.append(header);
    const selected = Object.create(null), openData = new Set(), controls = el('div',null,'controls'); controls.id = 'page-controls';
    const body = el('div'); body.id = 'page-content'; let version = 0, cleanup = [];
    function dispose(){cleanup.forEach(fn=>fn());cleanup=[];}
    function failed(error) {
      dispose();
      const message=el('p',config.errorText||error.message,'failure');message.setAttribute('role','alert');
      body.replaceChildren(message);console.error(error);
    }
    for (const control of config.controls || []) {
      selected[control.id] = control.value ?? control.options?.[0]?.value ?? '';
      const wrap = el('div',null,'control');
      if (control.type === 'tabs') {
        wrap.append(el('span',control.label)); const tabs = el('div',null,'tabs');
        for (const item of control.options) {
          const button = el('button',item.label); button.type='button';button.setAttribute('aria-pressed',String(item.value===selected[control.id]));
          button.onclick=()=>{selected[control.id]=item.value;for(const b of tabs.children)b.setAttribute('aria-pressed',String(b===button));update().catch(failed);};tabs.append(button);
        }
        wrap.append(tabs);
      } else {
        const label = el('label',control.label); label.htmlFor='control-'+control.id;
        const select = el('select'); select.id=label.htmlFor;
        for(const item of control.options){const option=el('option',item.label);option.value=item.value;select.append(option);}
        select.value=selected[control.id];select.onchange=()=>{selected[control.id]=select.value;update().catch(failed);};wrap.append(label,select);
      }
      controls.append(wrap);
    }
    if (controls.children.length) root.append(controls);
    root.append(body);
    async function update() {
      const current=++version;dispose();body.replaceChildren();
      const page = config.render ? config.render(CoworkSelection(selected)) : config;
      summary(body, page, 'page');
      for(const [index,s] of (page.sections||[]).entries()){
        const section=el('section',null,'section');section.id=s.id||'section-'+index;
        const head=el('div',null,'section-header');head.append(el('h2',s.title));if(s.caption)head.append(el('p',s.caption,'caption'));section.append(head);body.append(section);
        if(s.type==='text'){section.append(el('p',s.text,'text-block'));continue;}
        if(s.type==='table'){
          const wrap=el('div',null,'table-wrap'),table=el('table'),thead=el('thead'),tr=el('tr'),tbody=el('tbody');table.id='table-'+index;
          wrap.style.setProperty('--table-width',s.columns.length*104+'px');wrap.tabIndex=0;wrap.setAttribute('role','region');wrap.setAttribute('aria-label',s.title||'');
          for(const c of s.columns)tr.append(el('th',c.label));thead.append(tr);table.append(thead,tbody);
          for(const row of s.rows||[]){const tr=el('tr');for(const c of s.columns){const td=el('td');td.dataset.label=c.label;const raw=row[c.key],value=c.format?c.format(raw,row):CoworkNumber(raw,c.unit);td.append(el('span',value));tr.append(td);}tbody.append(tr);}
          if(!(s.rows||[]).length)section.append(el('p',config.emptyText||'—','empty'));else{wrap.append(table);section.append(wrap);}continue;
        }
        if(s.type==='timeline'){
          const list=el('ol',null,'timeline');for(const item of s.items||[]){const li=el('li');li.append(el('time',item.time),el('h3',item.label));if(item.detail)li.append(el('p',item.detail));list.append(li);}section.append(list);continue;
        }
        const view=el('div',null,s.type==='chart'?'view':'view geo-view');section.append(view);
        view.setAttribute('aria-busy','true');view.append(el('p',config.loadingText||'…','loading'));
        let chart;
        if(s.type==='chart')chart=await drawChart(view,s,cleanup);
        else if(s.type==='map'||s.type==='globe')await drawGeography(view,s,cleanup);
        else throw new Error('Unsupported section type: '+s.type);
        if(!view.isConnected||current!==version)return;
        if(!view.querySelector('canvas,svg'))throw new Error('The requested view did not draw.');
        view.setAttribute('aria-busy','false');
        if(chart&&window.CoworkChartData)CoworkChartData(chart,head,section,config,section.id,openData,cleanup);
        if(current!==version)return;
      }
    }
    try{await update();}catch(error){dispose();throw error;}
  };
})();
