"""Terminal implementation of :class:`Elicitor` for standalone CLI runs.

Rendering is split by question kind, on purpose:

* ``kind="choice"`` — the numbered list is drawn by ``StreamDisplay`` from
  the published ``StreamAskUser`` event, so this class only reads stdin.
  That keeps rendering in the rendering layer, and it is also what lets a
  GUI host show the same question as buttons.
* ``kind="path"`` — cowork has no file-browser widget, so path questions are
  not published and this class draws them itself.
"""

from __future__ import annotations

from typing import Awaitable, Callable

from rich.markup import escape

from anton.core.interaction.elicit import AskAnswer, AskRequest

__all__ = ["CLIElicitor"]


class CLIElicitor:
    """Prompts on the terminal (standalone ``anton`` chat)."""

    answer_hint = (
        "The user answers on a terminal: they send the number of an option, a "
        "comma-separated list of numbers for a multi-select, or type their own "
        "answer instead."
    )
    # stdin blocks until the human acts, matching how select_path has always
    # behaved. Autonomous runs withhold ask_user rather than time out.
    timeout_s = None

    def __init__(
        self,
        console,
        supported_kinds: tuple[str, ...] = ("choice", "path"),
        before_prompt: Callable[[str, AskRequest], Awaitable[None]] | None = None,
    ) -> None:
        self._console = console
        self.supported_kinds = tuple(supported_kinds)
        # Awaited right before prompt_toolkit takes the terminal, with the
        # question's id and request. `elicit()` only enqueues events onto an
        # out-of-band queue (StreamTaskProgress(phase="interactive") to stop
        # chat_ui's spinner, then StreamAskUser to print the question) —
        # draining that queue is a separate task and can still be running
        # when this method is called. prompt_toolkit's own first render is
        # synchronous (application.py: _redraw() before the first real
        # await), so without waiting here it can land before the spinner has
        # stopped or before the question has printed, corrupting or
        # reordering the terminal output. The wired callback (see chat.py)
        # stops the spinner directly and, for a published ("choice") request,
        # waits for confirmation that StreamAskUser was actually rendered.
        self.before_prompt = before_prompt

    async def begin(self, question_id: str, request: AskRequest) -> None:
        return None

    async def end(self, question_id: str) -> None:
        return None

    async def ask(self, question_id: str, request: AskRequest) -> AskAnswer:
        if self.before_prompt is not None:
            await self.before_prompt(question_id, request)
        if request.kind == "path":
            return await self._ask_path(request)
        return await self._ask_choice(request)

    # ── choice — the list is already on screen ───────────────────────
    async def _ask_choice(self, request: AskRequest) -> AskAnswer:
        """Read the reply. The options were already printed by StreamDisplay.

        Input is a comma-separated list of offered numbers, the value of an
        option typed directly, or anything else, which is taken as the
        user's own wording. With ``select="one"`` a list keeps only the
        first number — deliberately, so a stray "2,1" answers with 2 rather
        than erroring or silently answering with both.

        ``request.default_value`` (an option's ``value``) is forwarded as
        ``prompt_or_cancel``'s own ``default`` — that function already
        substitutes it for a blank Enter and shows it in the prompt's
        suffix, so a blank Enter and typing the value by hand arrive here
        as the same string and need no separate handling.

        ``request.compact`` drops the descriptive caption in favour of a
        bare input point — ``prompt_or_cancel`` still shows the default in
        its own suffix (e.g. ``(accept):``), so the hint survives even with
        no label text; pairs with ``StreamDisplay.show_question`` skipping
        the numbered list for the same request.
        """
        from anton.utils.prompt import prompt_or_cancel

        if request.compact:
            label = ""
        else:
            label = (
                "Send the numbers (comma-separated) or type your own answer"
                if request.select == "many"
                else "Send the answer number or type your own"
            )
        raw = await prompt_or_cancel(label, default=request.default_value)
        if raw is None or not raw.strip():
            return AskAnswer(status="cancelled")
        raw = raw.strip()

        values = {option.value for option in request.options}
        if raw in values:
            return AskAnswer(status="answered", values=(raw,))

        numbers = {str(i): option for i, option in enumerate(request.options, start=1)}
        tokens = [t.strip() for t in raw.split(",") if t.strip()]
        if tokens and all(token in numbers for token in tokens):
            picked = tuple(numbers[token].value for token in tokens)
            if request.select == "one":
                picked = picked[:1]
            return AskAnswer(status="answered", values=picked)
        # Anything that is not a clean list of offered numbers, and not one
        # of the option values typed directly, is the user answering in
        # their own words.
        return AskAnswer(status="answered", text=raw)

    # ── path — this class renders as well ────────────────────────────
    async def _ask_path(self, request: AskRequest) -> AskAnswer:
        from anton.utils.prompt import prompt_or_cancel

        # Browse mode has no visual file tree on a terminal — fall back to a
        # typed path (the GUI host gets a real navigable browser instead).
        if request.path_mode == "browse":
            self._console.print(f"\n[bold]{escape(request.prompt)}[/]")
            if request.root:
                self._console.print(f"  [dim]starting at {escape(request.root)}[/]")
            chosen = await prompt_or_cancel("Enter a path (Esc to cancel)")
            chosen = (chosen or "").strip()
            return (
                AskAnswer(status="answered", values=(chosen,))
                if chosen
                else AskAnswer(status="cancelled")
            )

        options = request.options
        if not options:
            return AskAnswer(status="cancelled")

        # Escaped: these come from the filesystem, so a directory named
        # "[dim]" would otherwise be swallowed as a Rich tag, and one
        # containing "[/]" would raise MarkupError out of ask() while the tool
        # is mid-dispatch.
        self._console.print(f"\n[bold]{escape(request.prompt)}[/]")
        for index, option in enumerate(options, start=1):
            icon = "📁" if option.kind == "folder" else "📄"
            detail = f"  [dim]{escape(option.detail)}[/]" if option.detail else ""
            self._console.print(
                f"  [bold]{index}[/]. {icon} {escape(option.label)}{detail}"
            )

        choice = await prompt_or_cancel(
            "Select a number (Esc to cancel)",
            choices=[str(i) for i in range(1, len(options) + 1)],
        )
        if choice is None:
            return AskAnswer(status="cancelled")
        try:
            selected = int(choice) - 1
        except ValueError:
            return AskAnswer(status="cancelled")
        if not 0 <= selected < len(options):
            return AskAnswer(status="cancelled")
        return AskAnswer(status="answered", values=(options[selected].value,))
