/* Explicit legends and requested geographic motion; source coordinates stay unchanged. */
(() => {
  window.CoworkGeoCenter = points => {
    if(!points.length)return undefined;
    let x=0,y=0,z=0;
    for(const point of points){const lat=point.lat*Math.PI/180,lon=point.lon*Math.PI/180;x+=Math.cos(lat)*Math.cos(lon);y+=Math.cos(lat)*Math.sin(lon);z+=Math.sin(lat);}
    if(Math.hypot(x,y,z)<1e-8)return [points[0].lon,points[0].lat];
    return [Math.atan2(y,x)*180/Math.PI,Math.atan2(z,Math.hypot(x,y))*180/Math.PI];
  };
  window.CoworkGeoLegend = (section, source) => {
    if (!source.legend?.length) return;
    const legend = document.createElement('ul'); legend.className = 'geo-legend';
    for (const item of source.legend) {
      if (typeof item.label !== 'string' || !CSS.supports('color', item.color)) {
        throw new TypeError('A geographic legend needs a text label and a valid color matching the plotted marks.');
      }
      const entry = document.createElement('li'), swatch = document.createElement('span'), label = document.createElement('span');
      swatch.className = 'legend-key'; swatch.style.backgroundColor = item.color; swatch.setAttribute('aria-hidden', 'true');
      label.textContent = item.label; entry.append(swatch, label); legend.append(entry);
    }
    section.append(legend);
  };
  window.CoworkGeoMotion = (node, source, cleanup, moved, globe) => {
    for (const name of ['routeMotion', 'autoRotate']) if (source[name] != null && typeof source[name] !== 'boolean') {
      throw new TypeError(name + ' must be a boolean.');
    }
    if (globe && source.routeMotion) throw new TypeError('Animated route flow uses the D3 map. A globe can use autoRotate for a turning geographic view.');
    if (matchMedia('(prefers-reduced-motion: reduce)').matches) return;
    if (source.routeMotion) for (const path of node.querySelectorAll('path.route')) path.classList.add('route-motion');
    if (!globe) return;
    const controls = globe.controls(); controls.autoRotate = !!source.autoRotate;
    if (source.autoRotate) {
      const speed = source.rotationSpeed ?? .5;
      if (typeof speed !== 'number' || !Number.isFinite(speed) || Math.abs(speed) > 4) {
        throw new TypeError('Globe rotationSpeed must be a finite number between -4 and 4.');
      }
      controls.autoRotateSpeed = speed;
    }
    const camera = globe.camera(); let position, rotation;
    const changed = () => {
      const nextPosition = camera.position.toArray(), nextRotation = camera.quaternion.toArray();
      if (nextPosition.some((value, i) => Math.abs(value - position[i]) > 1e-5)
          || nextRotation.some((value, i) => Math.abs(value - rotation[i]) > 1e-5)) moved();
      position = nextPosition; rotation = nextRotation;
    };
    // Camera initialization is not ongoing motion. Observe after the initial layout frames.
    let frame=requestAnimationFrame(()=>{frame=requestAnimationFrame(()=>{
      position=camera.position.toArray();rotation=camera.quaternion.toArray();
      controls.addEventListener('change', changed);
    });});
    cleanup.unshift(() => {cancelAnimationFrame(frame);controls.removeEventListener('change', changed);});
  };
  function placeLabels(points, projection, width, height, measure) {
    const placed=[];
    for(const point of points.filter(point=>point.label)) {
      const [px,py]=projection([point.lon,point.lat]);
      const length=measure(point),preferredX=px+8,preferredY=py-8;
      let best;
      for(let offset=0;offset<height;offset+=18) for(const direction of offset?[1,-1]:[1]) {
        const y=Math.max(14,Math.min(height-8,preferredY+direction*offset));
        for(const candidateX of [preferredX,px-length-8,8,width-length-8]) {
          const x=Math.max(8,Math.min(width-length-8,candidateX));
          const overlap=placed.reduce((area,box)=>area+Math.max(0,Math.min(x+length+5,box.x+box.w+5)-Math.max(x-5,box.x-5))*Math.max(0,16-Math.abs(y-box.y)),0);
          const distance=(x-preferredX)**2+(y-preferredY)**2;
          const score=overlap*1e6+distance;
          if(!best||score<best.score)best={x,y,score};
        }
      }
      placed.push({...best,w:length,point,px,py});
    }
    return placed;
  }
  window.CoworkMapLabels = (group, points, projection, width, height) => {
    const texts=new Map();
    const placements=placeLabels(points,projection,width,height,point=>{
      const text=group.append('text').text(point.label).attr('font-size',11).attr('fill','#e6edf3').attr('stroke','#101923').attr('stroke-width',3).attr('paint-order','stroke');
      texts.set(point,text);return text.node().getComputedTextLength();
    });
    for(const {point,px,py,x,y} of placements){
      const text=texts.get(point);
      group.insert('line',()=>text.node()).attr('x1',px).attr('y1',py).attr('x2',x).attr('y2',y-4).attr('stroke','#8ba7bc').attr('stroke-width',.7);
      text.attr('x',x).attr('y',y);
    }
  };
  window.CoworkGlobeLabels = (node, globe, points, cleanup) => {
    const overlay=document.createElement('div');overlay.className='point-annotations';node.append(overlay);
    function draw(){
      if(!node.isConnected)return;
      const width=node.clientWidth,height=node.clientHeight,view=globe.pointOfView();
      const lat=view.lat*Math.PI/180,lon=view.lng*Math.PI/180;
      const visible=points.filter(point=>{
        const p=point.lat*Math.PI/180,l=point.lon*Math.PI/180;
        return Math.sin(p)*Math.sin(lat)+Math.cos(p)*Math.cos(lat)*Math.cos(l-lon)>1/(1+view.altitude)+.02;
      });
      overlay.replaceChildren();const labels=new Map();
      const project=pair=>{const p=globe.getScreenCoords(pair[1],pair[0],.02);return [p.x,p.y];};
      const placements=placeLabels(visible,project,width,height,point=>{
        const label=document.createElement('span');label.className='point-note';label.textContent=point.label;overlay.append(label);labels.set(point,label);return label.getBoundingClientRect().width;
      });
      for(const {point,px,py,x,y} of placements){
        const label=labels.get(point);label.style.left=x+'px';label.style.top=y-12+'px';
        const leader=document.createElement('span');leader.className='point-leader';leader.setAttribute('aria-hidden','true');
        Object.assign(leader.style,{left:px+'px',top:py+'px',width:Math.hypot(x-px,y-4-py)+'px',transform:`rotate(${Math.atan2(y-4-py,x-px)}rad)`});overlay.prepend(leader);
      }
    }
    let frame;
    const schedule=()=>{if(frame)return;frame=requestAnimationFrame(()=>{frame=undefined;draw();});};
    const controls=globe.controls();controls.addEventListener('change',schedule);
    const observer=new ResizeObserver(schedule);observer.observe(node);schedule();
    cleanup.unshift(()=>{cancelAnimationFrame(frame);observer.disconnect();controls.removeEventListener('change',schedule);overlay.remove();});
  };
})();
