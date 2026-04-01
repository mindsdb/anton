from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from prompt_toolkit import PromptSession
    from rich.console import Console


@dataclass
class PromptSuggestion:
    """A single onboarding suggestion."""

    display_text: str
    prompt_text: str
    category: str


def build_suggestions() -> list[PromptSuggestion]:
    """Return a list of onboarding suggestions to show to the user on first run."""
    suggestions: list[PromptSuggestion] = []

    suggestions.append(
        PromptSuggestion(
            display_text=(
                "Build me an interactive AI stocks bracket dashboard — auto-pick the top 16 "
                "AI stocks by market cap, seed them, score each head-to-head matchup using a "
                "composite score of YTD return, revenue growth, and analyst ratings, and let "
                "me click to advance winners through the rounds."
            ),
            prompt_text=(
                "Build me an interactive AI stocks bracket dashboard — auto-pick the top 16 "
                "AI stocks by market cap, seed them, score each head-to-head matchup using a "
                "composite score of YTD return, revenue growth, and analyst ratings, and let "
                "me click to advance winners through the rounds."
            ),
            category="showcase",
        )
    )

    suggestions.append(
        PromptSuggestion(
            display_text=(
                "Explore a connected datasource and explain its schema, "
                "key tables, and relationships"
            ),
            prompt_text=(
                "Help me connect to a datasource. Walk me through the options, then once "
                "connected show me the schema, key tables, and explain the relationships "
                "between them."
            ),
            category="datasource",
        )
    )
    suggestions.append(
        PromptSuggestion(
            display_text=(
                "Analyze a dataset and surface key insights, trends, and anomalies"
            ),
            prompt_text=(
                "I'd like to analyze a dataset. I'll upload a CSV file — once you have it, "
                "surface the key insights, trends, distributions, and any anomalies worth "
                "investigating."
            ),
            category="analysis",
        )
    )

    suggestions.append(
        PromptSuggestion(
            display_text="Build a dashboard from live or connected data",
            prompt_text=(
                "Build me a dashboard from live or connected data. Either use a datasource "
                "I've connected, or pull from a public API (weather, crypto, stocks, GitHub "
                "stats) and display it in a clean terminal dashboard that auto-refreshes."
            ),
            category="dashboard",
        )
    )

    return suggestions


def display_suggestions(console: Console, suggestions: list[PromptSuggestion]) -> None:
    """Render the numbered suggestion list to the terminal using Rich markup."""
    console.print("[anton.prompt]anton>[/] You're all set 🎉")
    console.print()
    console.print("Here are a few things you can try:")
    console.print()
    for i, suggestion in enumerate(suggestions, 1):
        if suggestion.category == "showcase":
            console.print(f"{i}. [anton.cyan]{suggestion.display_text}[/]")
        else:
            console.print(f"{i}. [anton.cyan]\\[{suggestion.display_text}][/]")
        console.print()


async def prompt_for_selection(
    console: Console,
    suggestions: list[PromptSuggestion],
    prompt_session: PromptSession,
) -> str | None:
    """Prompt the user to pick a suggestion or type their own question.

    Returns:
        The prompt text to send to Anton, or None if the user skips (empty Enter).
        A number selects the corresponding suggestion; any other text is returned as-is.
    """
    options = "/".join(str(i) for i in range(1, len(suggestions) + 1))
    try:
        raw = await prompt_session.prompt_async(
            [("bold", f"Select [{options}] or type your own question: ")]
        )
    except (EOFError, KeyboardInterrupt):
        return None

    stripped = raw.strip()
    if not stripped:
        return None

    if stripped.isdigit():
        index = int(stripped) - 1
        if 0 <= index < len(suggestions):
            return suggestions[index].prompt_text
        return stripped

    return stripped
