/* Artifact Kit runtime: slide decks. Slides are 16:9 frames stacked in one scrolling page (easy to
   read, print and share); arrow keys and the pill step through them. Phones drop the fixed ratio. */
(function () {
  'use strict';
  var AK = window.AK;
  AK.deck = function (spec, mount) {
    var deck = AK.el('main', 'ak-deck');
    mount.appendChild(deck);
    var slides = spec.slides || [];
    var frames = [];
    slides.forEach(function (s, i) {
      var layout = s.layout || (i === 0 ? 'title' : 'content');
      var f = AK.el('section', 'ak-slide ak-slide-' + layout + ' ak-anim-in');
      f.setAttribute('aria-label', AK.t('slide') + ' ' + (i + 1));
      f.id = s.id || ('slide-' + (i + 1));
      if (layout === 'title' || layout === 'section') {
        if (s.eyebrow) f.appendChild(AK.el('div', 'ak-chip', s.eyebrow));
        f.appendChild(AK.el('h1', 'ak-title', s.title || ''));
        if (s.subtitle) { var st = AK.el('p', 'ak-subtitle'); st.innerHTML = AK.inline(s.subtitle); f.appendChild(st); }
        if (layout === 'section') f.classList.add('ak-slide-title');
      } else {
        var head = AK.el('div', 'ak-slide-head');
        if (s.title) head.appendChild(AK.el('h2', null, s.title));
        if (s.subtitle) { var p = AK.el('p'); p.innerHTML = AK.inline(s.subtitle); head.appendChild(p); }
        f.appendChild(head);
        var body = AK.el('div', 'ak-slide-body');
        f.appendChild(body);
        if (layout === 'big_number') {
          body.appendChild(AK.el('div', 'ak-big-number', s.value_text || AK.fmt(s.value, s.format || 'auto')));
          if (s.caption) { var cap = AK.el('div', 'ak-text ak-lead'); cap.innerHTML = AK.md(s.caption); body.appendChild(cap); }
          AK.renderAll(s.blocks, body);
        } else if (layout === 'split') {
          var g = AK.el('div', 'ak-grid');
          g.style.setProperty('--ak-cols', 2);
          var l = AK.el('div', 'ak-stack'), r = AK.el('div', 'ak-stack');
          g.appendChild(l); g.appendChild(r); body.appendChild(g);
          AK.renderAll(s.blocks, l); AK.renderAll(s.right, r);
        } else {
          AK.renderAll(s.blocks, body);
        }
      }
      var n = AK.el('div', 'ak-slide-num', (i + 1) + ' / ' + slides.length);
      f.appendChild(n);
      deck.appendChild(f);
      frames.push(f);
    });
    AK.footer(spec, deck);
    if (frames.length > 1) {
      var nav = AK.el('nav', 'ak-deck-nav');
      var prev = AK.el('button', null, '‹'), next = AK.el('button', null, '›'), label = AK.el('span', null, '1 / ' + frames.length);
      prev.type = next.type = 'button';
      prev.setAttribute('aria-label', AK.t('prev')); next.setAttribute('aria-label', AK.t('next'));
      nav.appendChild(prev); nav.appendChild(label); nav.appendChild(next);
      document.body.appendChild(nav);
      var cur = 0;
      var go = function (i) { cur = Math.max(0, Math.min(frames.length - 1, i)); frames[cur].scrollIntoView({behavior: 'smooth', block: 'center'}); label.textContent = (cur + 1) + ' / ' + frames.length; };
      prev.addEventListener('click', function () { go(cur - 1); });
      next.addEventListener('click', function () { go(cur + 1); });
      document.addEventListener('keydown', function (e) {
        if (e.target && /INPUT|SELECT|TEXTAREA/.test(e.target.tagName)) return;
        if (e.key === 'ArrowDown' || e.key === 'ArrowRight' || e.key === 'PageDown' || e.key === ' ') { e.preventDefault(); go(cur + 1); }
        if (e.key === 'ArrowUp' || e.key === 'ArrowLeft' || e.key === 'PageUp') { e.preventDefault(); go(cur - 1); }
      });
      if (window.IntersectionObserver) {
        var io = new IntersectionObserver(function (entries) {
          entries.forEach(function (en) { if (en.isIntersecting && en.intersectionRatio > 0.55) { cur = frames.indexOf(en.target); label.textContent = (cur + 1) + ' / ' + frames.length; } });
        }, {threshold: [0.55]});
        frames.forEach(function (f) { io.observe(f); });
      }
    }
  };
})();
