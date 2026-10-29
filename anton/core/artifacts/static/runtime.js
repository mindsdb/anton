(() => {
  'use strict';
  const outline = document.querySelector('#_artifact-outline');
  document.querySelector('[data-action="outline"]').addEventListener('click', event => {
    const open = outline.classList.toggle('open');
    event.currentTarget.setAttribute('aria-expanded', String(open));
  });
  const outlineButton=document.querySelector('[data-action="outline"]');
  outline.querySelectorAll('a').forEach(link=>link.addEventListener('click',()=>{outline.classList.remove('open');outlineButton.setAttribute('aria-expanded','false');}));
  const outlineObserver=new IntersectionObserver(entries=>{for(const entry of entries){if(entry.isIntersecting)outline.querySelectorAll('a').forEach(link=>link.setAttribute('aria-current',String(link.hash===`#${entry.target.id}`)));}}, {rootMargin:'-10% 0px -70% 0px'});
  document.querySelectorAll('#_artifact-content > .section').forEach(section=>outlineObserver.observe(section));
  document.querySelector('[data-action="print"]').addEventListener('click', () => window.print());
  document.querySelectorAll('[data-action="outline"],[data-action="print"]').forEach(button=>button.hidden=false);
  // Charts, maps and their labels are already rendered in the saved HTML.
  // JavaScript only enhances the complete, open-by-default data inspector.
  document.querySelectorAll('.chart[data-chart],.geo-map[data-native-map]').forEach(container=>{
    if(container.matches('.chart')&&!container.nextElementSibling?.matches('script.chart-spec'))return;
    const details=container.parentElement.querySelector('.chart-data');
    if(details)details.open=false;
  });
  document.querySelectorAll('.artifact-tabs').forEach(group=>{
    const list=group.querySelector('.tab-list');
    const tabs=[...list.children];
    const panels=[...group.querySelector('.tab-panels').children];
    const select=(index,focus=false)=>{
      tabs.forEach((tab,n)=>{tab.setAttribute('aria-selected',String(n===index));tab.tabIndex=n===index?0:-1;});
      panels.forEach((panel,n)=>panel.hidden=n!==index);
      if(focus)tabs[index].focus();
    };
    list.setAttribute('role','tablist');
    tabs.forEach((tab,n)=>{
      tab.setAttribute('role','tab');
      panels[n].setAttribute('role','tabpanel');panels[n].setAttribute('aria-labelledby',tab.id);panels[n].tabIndex=0;
      tab.addEventListener('click',()=>select(n));
      tab.addEventListener('keydown',event=>{
        const keys={ArrowRight:(n+1)%tabs.length,ArrowLeft:(n+tabs.length-1)%tabs.length,Home:0,End:tabs.length-1};
        if(Object.hasOwn(keys,event.key)){event.preventDefault();select(keys[event.key],true);}
      });
    });
    select(Number(group.dataset.initial));group.classList.add('tabs-enhanced');list.hidden=false;
  });
})();
