# Browser contract fixtures

These files pin the HTTP contract between Anton's `browser` tool and the two hosts that serve it: the MindsHub browser router (with the browser it starts for each person) and the Cowork desktop app's own browser. They also pin the turn request's browser block, `BrowserConfig`, and the session event the cloud turn writes.

Anton owns them. `tests/test_browser_contract.py` reads every file through typed models, parses each answer into the model in `anton/core/browser/models.py` that the client reads it with, and checks each request against what the client sends. Each host's tests keep a byte-for-byte copy of this folder, this README included, and check their own routes against it, so a change here is a contract change for both hosts, and a host keeps its own notes outside its copy.

## Format

Each file is one JSON object:

- `description`: what the file covers, in a sentence or two.
- `limits` (optional): named values both hosts hold the same: a number, such as `saved_tabs_per_project`, a regular expression's source, such as `card_field_name`, which a host compares with its own pattern's source and reads ignoring case, or a list, such as `payment_frame_hosts`. Each host compares its own constant with it. `card_field_word_breaks` is a list of regular expressions' sources, which a host compares with its own and reads case-sensitive.
- `examples`: named examples. Each one has:
  - `where`: the route (`GET /sessions/{id}/page`), or where the value goes (`params.browser`, `cloud turn stdout`).
  - `kind`: `request` (what Anton sends), `response` (what a host answers), `event` (what the cloud turn writes), `turn_block` (a turn request's `params.browser`), `constructor` (the arguments a host passes to `BrowserConfig` in its own process) or `classification` (a step on a page, and whether it waits for the person's approval, or a field's attribute values, and whether they name the field for a card).
  - `status`: the HTTP status, for a response.
  - `refused`: `true` when the receiver must refuse this body. A host answers a refused request with an error, and Anton never sends one. `BrowserConfig.from_dict` returns None for a refused turn block, and the constructor raises for refused arguments.
  - `body`: the JSON value itself.

## Files

| File | Covers |
|---|---|
| `sessions.json` | Opening a conversation's session, and the one answer shape both hosts give |
| `page.json` | The page snapshot: elements, `value_hidden` on secret fields, `focus` for the focused node, `challenges` for visible check frames, `frames` for child frames, and the card-field rule both hosts hold |
| `action.json` | Step requests, the focused steps both hosts refuse, and the action answer |
| `control.json` | The control record (`holder`, `reason`, `since`, `seq`, `by`), the hand-over request, and the refused take-back |
| `tabs.json` | Tab list, open, activate and close |
| `sites.json` | Signed-in sites the browser records |
| `saved_tabs.json` | A project's saved tabs, and the most a project keeps |
| `commit_steps.json` | The commit list: which steps wait for the person's approval, by commit or by position |
| `error.json` | Every error body, named by its code; status and code are the contract, the messages are examples |
| `session_event.json` | The cloud turn's `{"kind": "browser", "session_id"}` line |
| `browser_config.json` | Router turn blocks, the in-app browser's constructor arguments, and the forms both refuse |
