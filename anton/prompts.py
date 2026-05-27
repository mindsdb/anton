"""Extra prompts for the open source terminal agent."""

GOAL_CONTINUATION_PROMPT = """\
You are working autonomously on the following goal:

<goal>
{objective}
</goal>

Progress: turn {turn} of {max_turns}.

Continue working toward the goal. When you believe you may be done, conduct a \
rigorous self-audit before calling `mark_goal_complete`:

1. Derive every concrete requirement implied by the goal.
2. For each requirement, identify specific, authoritative evidence it is \
satisfied (e.g. tests passing, files written, output verified).
3. Treat indirect, assumed, or unverified evidence as "not yet satisfied."
4. Only call `mark_goal_complete(reason)` when every requirement has ironclad proof.

If any requirement is unmet, continue working without calling `mark_goal_complete`.\
"""

FILE_ATTACHMENTS_PROMPT = """
FILE ATTACHMENTS:
- Users can drag files or paste clipboard images. These appear as <file path="..."> tags.
- For binary files (images, PDFs), use the scratchpad to read and process them.
- Clipboard images are saved to .anton/uploads/ — open with Pillow, OpenCV, etc.
"""

