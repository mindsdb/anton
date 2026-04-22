"""
Format the Memory Access Log tab in the Session Sharing Feature Spec doc.
Reflects what was actually implemented in feature/memory-session-id.
"""
from doc_formatter import DocWriter, get_docs_client, find_tab

DOC_ID = "11XAI2Gmu43MJBiIseRGtCrYxV6FQncxPqlVvTCMqcfs"
TAB_TITLE = "Memory Access Log"


def main():
    docs = get_docs_client()
    doc = docs.documents().get(documentId=DOC_ID, includeTabsContent=True).execute()

    tab_id, _ = find_tab(doc, TAB_TITLE)
    if not tab_id:
        print(f"Tab '{TAB_TITLE}' not found.")
        return

    writer = DocWriter(tab_id)

    writer.title("Memory Access Log — Design Doc")
    writer.subtitle("Author: Alejandro Cantu + Claude Code  ·  April 2026  ·  Status: Implemented — branch feature/memory-session-id")
    writer.spacer()

    writer.heading1("Purpose")
    writer.body(
        "The memory access log is a lightweight, append-only, system-written record of "
        "which project memories were delivered to Anton's context during each session. "
        "It serves two purposes:"
    )
    writer.bullet([
        "Export accuracy — at /share export time, the access log is the ground truth for which project memories should travel with the .anton file. No LLM inference needed — it's a direct lookup.",
        "Auditability — the log is permanent. You can always answer 'which memories were active in session X?' — not just at export time but months later.",
    ])
    writer.spacer()

    writer.heading1("Design Principle")
    writer.body(
        "The access log is written by the system at the moment of memory delivery — "
        "not by Anton, not by an LLM judgment call, and not inferred after the fact."
    )
    writer.spacer()
    writer.body(
        "This is the same principle as a file system or database access log: the read "
        "is recorded at the moment it happens, deterministically, by the layer that "
        "performs the read. No extra LLM calls, no latency, no reliance on Anton's "
        "behavior mid-response."
    )
    writer.spacer()
    writer.code(
        "Session starts → cortex.build_memory_context() loads memories\n"
        "  → delivers to Anton's context\n"
        "  → writes log entry   ← happens here, always"
    )
    writer.spacer()

    writer.heading1("File Location")
    writer.code("<project>/.anton/memory/access_log.jsonl")
    writer.spacer()
    writer.body(
        "Append-only JSONL (one JSON object per line). Never overwritten — only appended to. "
        "Created automatically on first write. Treated as empty on read if missing."
    )
    writer.spacer()

    writer.heading1("Log Entry Format")
    writer.code(
        '{\n'
        '  "session_id": "20260421_103200",\n'
        '  "memory_id": "m_3a7f92bc",\n'
        '  "memory_scope": "project",\n'
        '  "memory_kind": "always",\n'
        '  "memory_topic": "database-schema",\n'
        '  "delivered_at": "2026-04-21T10:32:00Z"\n'
        '}'
    )
    writer.spacer()
    writer.table(
        headers=["Field", "Type", "Description"],
        rows=[
            ["session_id", "string", "The session in which the memory was delivered"],
            ["memory_id", "string", "Unique identifier of the memory entry (e.g. m_3a7f92bc)"],
            ["memory_scope", "string", "project or global — global entries are logged but never exported"],
            ["memory_kind", "string", "always / never / when / lesson / profile — profile entries logged but never exported"],
            ["memory_topic", "string", "Topic tag of the memory (e.g. database-schema). Empty string if none."],
            ["delivered_at", "ISO timestamp", "When the memory was delivered to Anton's context"],
        ]
    )
    writer.spacer()
    writer.body(
        "No reason field — delivery is a system event, not an interpreted action. "
        "The reason is implicit: the memory was in scope and was loaded."
    )
    writer.spacer()

    writer.heading1("When the System Writes to the Log")
    writer.table(
        headers=["Event", "Log entry written"],
        rows=[
            ["Session starts, project memories loaded into context", "✅ One entry per memory delivered"],
            ["Mid-session, memories reloaded (next turn)", "✅ One entry per memory delivered"],
            ["Memory is written (new lesson/rule encoded)", "✅ Already tagged with session_id at write time — no separate log entry needed"],
            ["Memory exists but was never loaded in this session", "❌ No entry"],
        ]
    )
    writer.spacer()

    writer.heading1("What Was Implemented")
    writer.table(
        headers=["File", "Change"],
        rows=[
            ["core/memory/access_log.py", "NEW — AccessLog class: log_delivered() appends entries, get_session_entries() filters by session_id for export"],
            ["core/memory/hippocampus.py", "Added list_rule_records() and list_lesson_records(token_budget) — return the exact records delivered, mirrors recall_* budget logic"],
            ["core/memory/cortex.py", "Accept access_log in __init__. build_memory_context() calls _log_delivered() after each memory section that has content. Pass session_id through."],
            ["core/session.py", "Forward session_id to build_memory_context()"],
            ["chat.py", "Wire AccessLog(project_memory_dir) into Cortex at startup"],
            ["tests/test_access_log_integration.py", "NEW — 20 integration tests covering AccessLog write, get_session_entries, Hippocampus record listing, and full Cortex integration"],
        ]
    )
    writer.spacer()

    writer.heading1("Export Time Lookup")
    writer.body(
        "At /share export, the export logic queries the access log to find which project "
        "memories were delivered in the session:"
    )
    writer.code(
        "accessed_memories = [\n"
        "    entry for entry in access_log.get_session_entries(session_id)\n"
        "    if entry['memory_kind'] != 'profile'\n"
        "    and entry['memory_scope'] != 'global'\n"
        "]"
    )
    writer.spacer()
    writer.body(
        "The matching memory IDs are resolved against the memory files and included "
        "in the .anton bundle under memory.project_accessed."
    )
    writer.spacer()

    writer.heading1("Edge Cases")
    writer.table(
        headers=["Case", "Behavior"],
        rows=[
            ["Access log is empty for the current session", "Export proceeds with session-born memories only. Anton notes this in the export summary."],
            ["Same memory delivered multiple times in a session", "Multiple entries in the log for the same memory_id. Deduplicate at export time — only one copy travels in the .anton file."],
            ["Memory is deleted after being logged", "At export time, if a logged memory_id no longer exists in the memory files, it is silently skipped. The log entry remains for audit purposes."],
            ["Log file does not exist yet", "Created automatically on first write. Treated as empty on read if missing — no error."],
        ]
    )
    writer.spacer()

    writer.heading1("Testing")
    writer.body("Tested on branch feature/memory-session-id:")
    writer.bullet([
        "95 unit + integration tests passing (20 new tests for access log)",
        "access_log.jsonl created and appended to on every memory delivery in a live session",
        "Log entries verified: correct session_id, memory_id, scope, kind, topic, delivered_at",
        "Different sessions produce separate log entries — get_session_entries() filters correctly",
        "No entries written when session_id is None (episodic memory disabled)",
        "No entries written when access_log is not configured (backwards compatible)",
        "list_lesson_records() applies same token budget as recall_lessons() — exact delivery match",
    ])
    writer.spacer()

    writer.heading1("What Does NOT Change")
    writer.bullet([
        "Memory delivery to context (cortex.py output) — unchanged",
        "The /memorize and /recall tool interfaces — unchanged",
        "JSONL storage format for engrams — unchanged (see Memory JSONL Format tab)",
        "Global memory is logged but excluded from export per the Memory Export Filter spec",
    ])
    writer.spacer()

    writer.heading1("Future Extensions (Phase 2)")
    writer.bullet([
        "Access frequency analytics — which memories are delivered most often across sessions? Useful for identifying stale or redundant rules.",
        "Cross-session memory lineage — trace how a memory evolved: written in session A, delivered in sessions B and C, exported in session D.",
        "Memory relevance scoring — use delivery frequency as a signal for which memories matter most during context distillation.",
    ])

    writer.apply(docs, DOC_ID, clear=True)
    print("Memory Access Log tab updated successfully.")


if __name__ == "__main__":
    main()
