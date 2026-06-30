"""Anton analytical-quality evals (ENG-381).

A small, repeatable harness that runs Anton end-to-end on a handful of realistic
analytical tasks and scores the output, so we can measure analysis quality over
time — in particular, capture a baseline before the ENG-380 prompting fixes and
prove the lift afterwards.

This is a *quality* eval, not a unit-test suite: it drives a real
``ChatSession.turn()`` against a real model and grades the answer with a hybrid
of deterministic fact checks and an LLM judge. See ``README.md``.
"""
