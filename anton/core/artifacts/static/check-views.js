(() => {
  'use strict';
  const payload=JSON.parse(document.getElementById('_artifact-bindings').textContent);
  if(!payload.view_checks?.length)return;
  const verify=()=>setTimeout(()=>{
    for(const check of payload.view_checks){
      let found;
      try{found=[...document.querySelectorAll(check.selector)];}
      catch{throw new Error(`Invalid artifact view selector: ${check.selector}`);}
      const meaningful=found.filter(el=>el.textContent.trim() || el.matches('path,rect,circle,line,polygon,polyline,image,canvas') || el.querySelector('path,rect,circle,line,polygon,polyline,image,canvas,img'));
      if(meaningful.length<check.minimum){
        const message=`Artifact view ${check.selector}: expected at least ${check.minimum} data marks or records, found ${meaningful.length}.`;
        console.error(message);
        const alert=document.createElement('p');alert.setAttribute('role','alert');alert.textContent='This view did not finish rendering.';document.body.append(alert);
      }
    }
  },350);
  if(document.readyState==='loading')document.addEventListener('DOMContentLoaded',verify,{once:true});else verify();
})();
