(() => {
  'use strict';
  if (!document.documentElement.classList.contains('slides')) return;
  const root=document.documentElement;
  const slides=[...document.querySelectorAll('.section')];
  const chrome=[...document.querySelectorAll('.document-header,main>footer')];
  const present=document.querySelector('[data-action="present"]');
  present.hidden=false;
  const controls=document.createElement('div');controls.className='slide-controls';controls.hidden=true;
  controls.innerHTML='<button type="button" aria-label="Previous slide">←</button><span role="status" aria-live="polite"></span><button type="button" aria-label="Next slide">→</button>';
  document.querySelector('.toolbar').append(controls);
  const [previous,next]=controls.querySelectorAll('button');const status=controls.querySelector('span');
  let active=0, presenting=false;
  const fromHash=()=>Math.max(0,slides.findIndex(s=>s.id===location.hash.slice(1)));
  const show=index=>{
    active=Math.max(0,Math.min(slides.length-1,index));
    slides.forEach((slide,n)=>{slide.hidden=presenting && n!==active;});
    previous.disabled=active===0;next.disabled=active===slides.length-1;
    status.textContent=`${active+1} / ${slides.length}`;
    if(presenting){document.title=slides[active].querySelector('h1,h2')?.textContent || document.title;window.scrollTo(0,0);}
  };
  const originalTitle=document.title;
  const toggle=()=>{
    presenting=!presenting;root.classList.toggle('presenting',presenting);
    chrome.forEach(element=>element.hidden=presenting);
    controls.hidden=!presenting;present.textContent=presenting?'Read':'Present';
    present.setAttribute('aria-pressed',String(presenting));
    if(!presenting)document.title=originalTitle;
    show(active);
  };
  present.addEventListener('click',toggle);
  previous.addEventListener('click',()=>show(active-1));next.addEventListener('click',()=>show(active+1));
  document.addEventListener('keydown',event=>{
    if(!presenting || event.altKey || event.ctrlKey || event.metaKey || event.target.closest('input,textarea,select,button,summary,[contenteditable]'))return;
    const actions={ArrowRight:()=>show(active+1),ArrowDown:()=>show(active+1),' ':()=>show(active+1),ArrowLeft:()=>show(active-1),ArrowUp:()=>show(active-1),Home:()=>show(0),End:()=>show(slides.length-1),Escape:toggle};
    if(actions[event.key]){event.preventDefault();actions[event.key]();}
  });
  document.querySelectorAll('#_artifact-outline a').forEach(link=>link.addEventListener('click',()=>{const index=slides.findIndex(slide=>`#${slide.id}`===link.hash);if(index>=0)show(index);}));
  window.addEventListener('hashchange',()=>show(fromHash()));
  show(fromHash());
})();
