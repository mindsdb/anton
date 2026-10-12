---
name: browser
description: 'Recall BEFORE the first browser tool call in a task, and whenever a
  site needs a sign-in, a CAPTCHA, a code or a payment. How to work in the browser
  you share with the user: when to use it instead of a connector, how to split a
  task between run and act, when to hand over to the user, what to remember about
  their logins (never the credentials), and what must be confirmed first.'
metadata:
  display_name: Working in the shared browser
  provenance: builtin
---
THE SHARED BROWSER

The user's browser is a real Chromium they also see beside this chat. Treat it as a desk you share: they can click, type and scroll in it at any moment, and everything you do happens in front of them.

When to use it
- Use a connector or an API first when one can do the job. Reach for the browser when no connector covers the site, the task is visual or interactive, or the user asks.
- Open only sites the user named or that the task plainly needs. Never guess URLs for internal tools; ask.

How to drive it
1. Start with `run`: one sentence of goal, plus `inputs` for anything that must be typed or chosen ({"search": "MacBook Air", "country": "Mexico"}). run takes the steps it is sure of and hands back with the page and the steps taken when it isn't.
2. When run hands back, read the page and take the next step yourself with `act`, using the element numbers from the latest page. Then call run again if the rest is routine.
3. Element numbers can go stale after the page changes. If a step says an element is gone, read the page again rather than guessing.
4. Use `screenshot` only when the page list can't show what you need: canvas, maps, charts, images with text.
5. Report only what the page shows. If a step did nothing, say so; never describe a page you haven't read.

Working with the user
- Sign-ins, CAPTCHAs, two-factor codes, card details and anything else secret are the user's to do. Call `hand_to_user` with one line saying what's needed ("Please sign in to Salesforce"), and continue when they're done. Never ask them to type a password into the chat, and never type one yourself.
- Before anything that commits on their behalf (buying, paying, sending a message or email, deleting, submitting a form), ask first, with what will happen in plain words.
- If the user takes over mid-task, read the page again before your next step: they may have moved things on.
- For a long task the user delegated, say briefly what you're doing as you go, so the viewer and the chat tell the same story.

Remembering logins
- After the user signs in, call `note_login` with the site and an account hint they'd recognise (an email or workspace name, never a secret). It records the site in the browser and in memory, so the next conversation knows they're already signed in.
- `open` lists the sites on record and whether the browser still has cookies for each. A site marked expired needs a fresh sign-in: hand over instead of trying the old route.
- Memory says a login existed; the browser decides whether it still works. Check by opening the site, not by trusting the memory.
