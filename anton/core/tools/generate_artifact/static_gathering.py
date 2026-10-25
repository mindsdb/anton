"""Optional static-data collection with an explicit, checked completion condition."""
import copy
import os

from .observed_data import credential_bearing, read_handoff


def enabled(state):
    return (state.act_first and state.artifact_type == "html-app"
            and os.environ.get("ANTON_STATIC_ATOMIC_GATHER") == "on"
            and os.environ.get("ANTON_STATIC_DATA_BINDING") == "on"
            and os.environ.get("ANTON_STATIC_CANONICAL_ONLY") == "on")


def tools(schemas):
    result = copy.deepcopy(schemas)
    scratchpad = next(t for t in result if t["name"] == "scratchpad")
    scratchpad["input_schema"]["properties"]["complete_static_artifact"] = {
        "type": "object",
        "description": (
            "Optional: finish static evidence collection after this exec succeeds and writes a fresh, "
            "valid observed-data.json in the artifact folder. Use when the source and scope are "
            "resolved. Omit during exploration, user questions, or incomplete data collection."
        ),
        "properties": {
            "summary": {"type": "string", "description": "Short scope/source description; no guessed results."},
            "data_sources": {"type": "array", "items": {"type": "string"}},
            "assumptions": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["summary", "data_sources"],
    }
    scratchpad["description"] += (
        " For resolved static-artifact collection, include complete_static_artifact in the same "
        "exec call. Fetch and validate the requested data and save observed-data.json. The host "
        "completes collection only if execution succeeds and that fresh data file is valid."
    )
    return result


def fingerprint(folder):
    try:
        st = (folder / "observed-data.json").stat(follow_symlinks=False)
    except FileNotFoundError:
        return None
    return st.st_dev, st.st_ino, st.st_mtime_ns, st.st_ctime_ns, st.st_size


def validate_handoff(folder, before):
    after = fingerprint(folder)
    if after is None or after == before:
        raise ValueError("This execution did not write a fresh observed-data.json.")
    value = read_handoff(folder)
    if credential_bearing(value):
        raise ValueError("The data handoff contains credentials; save data without credentials.")
