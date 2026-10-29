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
  // Charts and their labels are already rendered in the saved HTML.
  // JavaScript only enhances the complete, open-by-default data inspector.
  document.querySelectorAll('.chart[data-chart]').forEach(container=>{
    if(!container.nextElementSibling?.matches('script.chart-spec'))return;
    const details=container.parentElement.querySelector('.chart-data');
    if(details)details.open=false;
  });
})();
