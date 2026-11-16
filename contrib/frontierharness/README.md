# FrontierHarness Eval adapter

Runs anton on [FrontierHarness Eval](https://github.com/frontier-harness-eval/eval) v1.0: 21
Terminal-Bench tasks through Harbor and 9 DeepSWE tasks through Pier, with the model fixed at
Kimi K3. The eval's own scripts provision the runtime, run the tasks and score them; this folder
plugs anton into them as a custom agent.

| File | Role |
| --- | --- |
| `anton_harbor.py`, `anton_pier.py` | Harbor and Pier agent classes |
| `anton_common.py` | What both share: install, run, token accounting |
| `install.sh` | The eval's `--install-script`: checks both runners can load the adapter |
| `run-task.sh` | The eval's `--cmd`: runs one task through Harbor or Pier |
| `eval-anton-usage.patch` | Teaches the eval's cost script to read anton's per-call usage |

## Run it with the eval's workflow on Runta

From a checkout of the eval at `e837a70bd6beb4e72eeeda62dd06e3bd34f6cb63`, with `ANTON` pointing
at an anton checkout and `SHA` the anton commit to evaluate:

```bash
git apply "$ANTON/contrib/frontierharness/eval-anton-usage.patch"
runta secret set MINDS_API_KEY --prompt
FH=skills/frontierharness-eval/scripts

bash $FH/provision-golden-checkpoint.sh \
  --runtime fh-build --checkpoint fh-golden-anton-$SHA --keep-runtime \
  --harness anton --repo https://github.com/mindsdb/anton --commit $SHA \
  --provider custom --model mindshub/kimi-k3 \
  --secret-name MINDS_API_KEY --secret-host api.mindshub.ai \
  --cpus 4 --memory 8192 --disk-size-gib 50 \
  --install-script "$ANTON/contrib/frontierharness/install.sh"

bash $FH/run-trials.sh \
  --checkpoint fh-golden-anton-$SHA --harness anton --run-id anton-$SHA --out runs \
  --provider custom --model mindshub/kimi-k3 \
  --secret-name MINDS_API_KEY --secret-host api.mindshub.ai \
  --cmd '/work/harness/contrib/frontierharness/run-task.sh {suite} {task} {model} {jobs}'

node $FH/normalize-results.mjs --run runs/anton-$SHA --label "anton $SHA"
node $FH/generate-chart.mjs --run runs/anton-$SHA
node $FH/build-report.mjs --run runs/anton-$SHA
```

`--model mindshub/kimi-k3` only labels the run in reports and pricing. The adapter asks the
MindsHub gateway for `kimi-k3`, which it serves with Fireworks Kimi K3, the provider of the
published runs.

To call Fireworks directly instead, store the key as `FIREWORKS_API_KEY` and provision a new
checkpoint with `--provider fireworks` in place of `--provider custom --model ... --secret-name
... --secret-host ...` (explicit flags override the preset, and the run must use the
checkpoint's provider). Run the trials the same way, with the `--cmd` prefixed by
`ANTON_FH_BASE_URL=https://api.fireworks.ai/inference/v1 ANTON_FH_MODEL=accounts/fireworks/models/kimi-k3`.
That route has not been run yet: MindsHub adjusts some tool schemas for Kimi K3 that raw
Fireworks may reject.

`run-task.sh` also runs on a laptop with Docker, Harbor 0.22.0 and Pier 0.3.1 installed as uv
tools: set `ANTON_FH_API_KEY` and `DEEP_SWE` (a deep-swe checkout at
`435ee89ec2f2e2289f33b0da4f992f0b7b7266b9`).

## What the adapter does

- Installs anton inside each task container from the GitHub tarball of the commit
  `run-task.sh` is checked out at, with uv's Python 3.12.
- Sends the task instruction as one non-interactive turn of `python -m anton.cloud_turn`, so
  anton never waits for a user.
- Turns off Jev, analytics and auto-updates, so every model call goes to Kimi K3.
- Keeps anton's `.anton/` folder out of git while the agent works and deletes it after, so it
  never reaches the graded diff.
- Exits 1 when the turn does not end in `turn_completed`; Harbor retries that twice, as for the
  published harnesses, and Pier does not retry.
- Reads token counts from anton's `llm_usage` log lines, including calls made from its
  scratchpad, and counts calls served by another model in `non_k3_calls`.

The adapter has run on local Docker but not on Runta yet. The first Runta run should confirm
that anton's Python trusts Runta's egress CA inside task containers, that the stub key is
swapped for the real one on the gateway host, and that Pier lets the agent reach that host.
