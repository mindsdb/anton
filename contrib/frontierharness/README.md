# FrontierHarness Eval adapter

Runs anton on [FrontierHarness Eval](https://github.com/frontier-harness-eval/eval) as a Harbor
agent. Version 1.1 is 62 tasks in one Harbor dataset on Runta runtimes; this folder plugs anton
into it, and into the older v1.0 workflow (Harbor and Pier).

| File | Role |
| --- | --- |
| `anton_harbor.py` | Harbor agent class: `-a anton_harbor:AntonHarbor` |
| `anton_pier.py` | Pier agent class, for the v1.0 DeepSWE tasks |
| `anton_common.py` | What both share: install, run, token accounting |
| `install.sh`, `run-task.sh` | v1.0 only: the eval's `--install-script` and `--cmd` |
| `eval-anton-usage.patch` | v1.0 only: lets the eval's cost script read anton's per-call usage |

## Run it on v1.1

From a checkout of the eval at v1.1 after `uv sync --locked`, with `ANTON` an anton checkout and
`SHA` the anton commit to evaluate (it must contain this folder):

```bash
export RUNTA_CONFIG=~/.config/runta/config.toml
export PYTHONPATH="$ANTON/contrib/frontierharness"
export ANTON_FH_REF=$SHA
export ANTON_FH_BASE_URL=https://api.mindshub.ai/v1 ANTON_FH_API_KEY=<MindsHub key>

uv run harbor run -d frontier-harness-eval/frontier-harness-eval@v1.1.0 -e runta \
  -a anton_harbor:AntonHarbor -m openai/kimi-k3 --override-storage-mb 32768
```

`-m` names the model as `provider/model`; anton asks its endpoint for the part after the slash.
In the eval's Baseten suite (`suites/baseten-eval.json`) every agent already gets the model proxy
as `OPENAI_BASE_URL` and `OPENAI_API_KEY`, which the adapter reads when the `ANTON_FH_*` values
are not set, so anton runs there as an agent entry with `"import_path":
"anton_harbor:AntonHarbor"`, `ANTON_FH_REF` in its `env`, and this folder on `PYTHONPATH`.

| Setting | Default | Purpose |
| --- | --- | --- |
| `ANTON_FH_REF` | required | anton commit installed in each task |
| `ANTON_FH_BASE_URL` | `OPENAI_BASE_URL` | OpenAI-compatible endpoint |
| `ANTON_FH_API_KEY` | `OPENAI_API_KEY` | Key for that endpoint |
| `ANTON_FH_MODEL` | from `-m` | Model id, if it differs from `-m` |
| `ANTON_FH_MAX_TOKENS` | `131072` | Longest reply, the cap v1.1 sets for every harness |

## Run it on v1.0

v1.0 (30 tasks, Kimi K3) lives at eval commit `e837a70bd6beb4e72eeeda62dd06e3bd34f6cb63`. Apply
`eval-anton-usage.patch` there, then use its `provision-golden-checkpoint.sh` with
`--install-script install.sh` and its `run-trials.sh` with
`--cmd '/work/harness/contrib/frontierharness/run-task.sh {suite} {task} {model} {jobs}'`.
`run-task.sh` also runs on a laptop with Docker, Harbor 0.22.0 and Pier 0.3.1, given
`ANTON_FH_API_KEY` and `DEEP_SWE` (a deep-swe checkout at
`435ee89ec2f2e2289f33b0da4f992f0b7b7266b9`).

## What the adapter does

- Installs anton inside each task from the GitHub tarball of `ANTON_FH_REF`, with uv's
  Python 3.12.
- Sends the task instruction as one non-interactive turn of `python -m anton.cloud_turn`, so
  anton never waits for a user.
- Turns off Jev (MindsHub's decision model, used for completion checks and the scratchpad's
  `classify()`), analytics and auto-updates, so every model call goes to the benchmark's model.
- Keeps anton's `.anton/` folder out of git while the agent works and deletes it after, so it
  never reaches the graded diff.
- Exits 1 when the turn does not end in `turn_completed`, so the runner treats a failed turn
  as a harness error.
- Reads token counts from anton's `llm_usage` log lines, including calls made from its
  scratchpad, and counts calls served by another model than the one asked for in
  `other_model_calls`.
