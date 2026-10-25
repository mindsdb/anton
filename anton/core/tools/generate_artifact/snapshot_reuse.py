"""Offer existing artifact evidence to presentation edits without bypassing collection."""
from anton.utils.nofollow import read_bytes_nofollow

from .observed_data import credential_bearing, identity, previous_data, read_handoff


def instructions(folder, primary):
    try:
        snapshot = read_handoff(folder)
        if credential_bearing(snapshot):
            return ""
        html = read_bytes_nofollow(folder, folder / primary, limit=8 * 1024 * 1024).decode("utf-8")
        embedded = previous_data(html)
        if identity(snapshot) not in embedded:
            return ""
    except (OSError, ValueError, RecursionError):
        return ""
    return (
        "\nExisting artifact snapshot: observed-data.json matches data already embedded in this "
        "same artifact's page. For a presentation-only edit or retry using the same facts and "
        "source/time scope, inspect and reuse this snapshot instead of fetching and recalculating "
        "it unnecessarily. Preserve its complete data, scope, units and provenance. This is not "
        "evidence of fresh service access. Requests for refreshed observations, new entities, "
        "different scopes or missing facts still require the relevant source work. Check whether "
        "the snapshot actually answers this request before reusing it. The existing successful "
        "execution and fresh-file validation still apply: read, validate and write the snapshot "
        "in the collection exec, even when its content remains unchanged."
    )
