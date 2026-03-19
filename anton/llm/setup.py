from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from rich.prompt import Prompt

from anton.llm.ollama import OllamaModelInfo, list_ollama_models, normalize_ollama_base_url

if TYPE_CHECKING:
    from rich.console import Console

    from anton.config.settings import AntonSettings
    from anton.workspace import Workspace


@dataclass(frozen=True)
class ProviderOption:
    key: str
    label: str
    badge: str
    default_planning_model: str
    default_coding_model: str


PROVIDER_OPTIONS: tuple[ProviderOption, ...] = (
    ProviderOption(
        key="anthropic",
        label="Anthropic (Claude)",
        badge="recommended",
        default_planning_model="claude-sonnet-4-6",
        default_coding_model="claude-haiku-4-5-20251001",
    ),
    ProviderOption(
        key="openai",
        label="OpenAI (GPT / o-series)",
        badge="experimental",
        default_planning_model="gpt-5-mini",
        default_coding_model="gpt-5-nano",
    ),
    ProviderOption(
        key="ollama",
        label="Ollama (local models)",
        badge="local",
        default_planning_model="",
        default_coding_model="",
    ),
    ProviderOption(
        key="openai-compatible",
        label="OpenAI-compatible (custom endpoint)",
        badge="experimental",
        default_planning_model="",
        default_coding_model="",
    ),
)


def configure_llm_settings(
    console,
    settings,
    workspace,
    *,
    show_current_config: bool = True,
) -> bool:
    """Prompt for provider/model configuration and persist it to the workspace."""
    if show_current_config:
        console.print()
        console.print("[anton.cyan]Current configuration:[/]")
        console.print(f"  Provider (planning): [bold]{settings.planning_provider}[/]")
        console.print(f"  Provider (coding):   [bold]{settings.coding_provider}[/]")
        console.print(f"  Planning model:      [bold]{settings.planning_model}[/]")
        console.print(f"  Coding model:        [bold]{settings.coding_model}[/]")
        console.print()
    else:
        console.print()
        console.print("[anton.cyan]LLM configuration[/]")
        console.print()

    option_by_number = {
        str(index): option
        for index, option in enumerate(PROVIDER_OPTIONS, start=1)
    }
    current_number = current_provider_number(settings.planning_provider)

    console.print("[anton.cyan]Available providers:[/]")
    for number, option in option_by_number.items():
        console.print(
            f"  [bold]{number}[/]  {option.label:<36} [dim]\\[{option.badge}][/]"
        )
    console.print()

    choice = Prompt.ask(
        "Select provider",
        choices=list(option_by_number),
        default=current_number,
        console=console,
    )
    provider = option_by_number[choice]

    if provider.key == "ollama":
        config = _prompt_for_ollama_config(console, settings, provider)
        if config is None:
            return False
        planning_model, coding_model = config
        settings.ollama_base_url = normalize_ollama_base_url(settings.ollama_base_url)
        workspace.set_secret("ANTON_OLLAMA_BASE_URL", settings.ollama_base_url)
    else:
        if provider.key == "openai-compatible":
            current_base_url = settings.openai_base_url or ""
            console.print()
            base_url = Prompt.ask(
                "API base URL [dim](e.g. http://localhost:11434/v1)[/]",
                default=current_base_url,
                console=console,
            ).strip()
            if base_url:
                settings.openai_base_url = base_url
                workspace.set_secret("ANTON_OPENAI_BASE_URL", base_url)

        api_key = _prompt_for_api_key(console, settings, provider.key)
        if api_key is None:
            return False

        planning_model, coding_model = _prompt_for_cloud_models(console, settings, provider)
        if api_key:
            key_name = api_key_env_name(provider.key)
            if key_name:
                workspace.set_secret(key_name, api_key)
                _set_provider_api_key(settings, provider.key, api_key)

    settings.planning_provider = provider.key
    settings.coding_provider = provider.key
    settings.planning_model = planning_model
    settings.coding_model = coding_model
    workspace.set_secret("ANTON_PLANNING_PROVIDER", provider.key)
    workspace.set_secret("ANTON_CODING_PROVIDER", provider.key)
    workspace.set_secret("ANTON_PLANNING_MODEL", planning_model)
    workspace.set_secret("ANTON_CODING_MODEL", coding_model)
    return True


def current_provider_number(provider: str) -> str:
    for index, option in enumerate(PROVIDER_OPTIONS, start=1):
        if option.key == provider:
            return str(index)
    return "1"


def api_key_env_name(provider: str) -> str | None:
    if provider == "anthropic":
        return "ANTON_ANTHROPIC_API_KEY"
    if provider in {"openai", "openai-compatible"}:
        return "ANTON_OPENAI_API_KEY"
    return None


def _prompt_for_api_key(console, settings, provider: str) -> str | None:
    key_attr = "anthropic_api_key" if provider == "anthropic" else "openai_api_key"
    current_key = getattr(settings, key_attr) or ""
    masked = _mask_secret(current_key) if current_key else "***"
    console.print()
    api_key = Prompt.ask(
        f"API key for {provider.title()} [dim](Enter to keep {masked})[/]",
        default="",
        console=console,
    ).strip()
    if api_key:
        return api_key
    if current_key:
        return ""
    console.print()
    console.print(f"[anton.error]No API key set for {provider}. Configuration not applied.[/]")
    console.print()
    return None


def _prompt_for_cloud_models(console, settings, provider: ProviderOption) -> tuple[str, str]:
    console.print()
    planning_model = Prompt.ask(
        "Planning model",
        default=(
            settings.planning_model
            if provider.key == settings.planning_provider
            else provider.default_planning_model
        ),
        console=console,
    )
    coding_model = Prompt.ask(
        "Coding model",
        default=(
            settings.coding_model
            if provider.key == settings.coding_provider
            else provider.default_coding_model
        ),
        console=console,
    )
    return planning_model, coding_model


def _prompt_for_ollama_config(console, settings, provider: ProviderOption) -> tuple[str, str] | None:
    current_url = settings.ollama_base_url or "http://localhost:11434"
    console.print()
    base_url = Prompt.ask(
        "Ollama URL [dim](e.g. http://localhost:11434)[/]",
        default=current_url,
        console=console,
    ).strip()
    settings.ollama_base_url = normalize_ollama_base_url(base_url or current_url)

    try:
        models = list_ollama_models(settings.ollama_base_url)
    except Exception as exc:
        console.print()
        console.print(
            "[anton.warning]Could not query Ollama for local models.[/]"
            f" [dim]({exc})[/]"
        )
        console.print("[anton.muted]Enter model names manually instead.[/]")
        console.print()
        planning_model = Prompt.ask(
            "Planning model",
            default=settings.planning_model if provider.key == settings.planning_provider else "",
            console=console,
        ).strip()
        if not planning_model:
            console.print("[anton.error]No model provided. Configuration not applied.[/]")
            console.print()
            return None
        coding_model = Prompt.ask(
            "Coding model",
            default=settings.coding_model if provider.key == settings.coding_provider else planning_model,
            console=console,
        ).strip()
        return planning_model, coding_model or planning_model

    if not models:
        console.print()
        console.print("[anton.warning]No local Ollama models found.[/]")
        console.print("[anton.muted]Pull a model with `ollama pull <model>` or enter a name manually.[/]")
        console.print()
        planning_model = Prompt.ask(
            "Planning model",
            default=settings.planning_model if provider.key == settings.planning_provider else "",
            console=console,
        ).strip()
        if not planning_model:
            console.print("[anton.error]No model provided. Configuration not applied.[/]")
            console.print()
            return None
        coding_model = Prompt.ask(
            "Coding model",
            default=settings.coding_model if provider.key == settings.coding_provider else planning_model,
            console=console,
        ).strip()
        return planning_model, coding_model or planning_model

    console.print()
    console.print("[anton.cyan]Available local Ollama models:[/]")
    for index, model in enumerate(models, start=1):
        console.print(f"  [bold]{index}[/]  {model.display_name}")
    manual_choice = str(len(models) + 1)
    console.print(f"  [bold]{manual_choice}[/]  Enter model name manually")

    planning_model = _prompt_for_ollama_model_choice(
        console=console,
        label="Planning model",
        models=models,
        default_model=(
            settings.planning_model
            if provider.key == settings.planning_provider
            else models[0].name
        ),
    )
    coding_model = _prompt_for_ollama_model_choice(
        console=console,
        label="Coding model",
        models=models,
        default_model=(
            settings.coding_model
            if provider.key == settings.coding_provider
            else planning_model
        ),
        manual_default=planning_model,
    )
    return planning_model, coding_model


def _prompt_for_ollama_model_choice(
    *,
    console,
    label: str,
    models: list[OllamaModelInfo],
    default_model: str,
    manual_default: str = "",
) -> str:
    choice_by_number = {
        str(index): model.name
        for index, model in enumerate(models, start=1)
    }
    manual_choice = str(len(models) + 1)
    choices = [*choice_by_number.keys(), manual_choice]
    default_choice = manual_choice
    for number, model_name in choice_by_number.items():
        if model_name == default_model:
            default_choice = number
            break

    choice = Prompt.ask(
        label,
        choices=choices,
        default=default_choice,
        console=console,
    )
    if choice == manual_choice:
        return Prompt.ask(
            f"{label} name",
            default=manual_default or default_model,
            console=console,
        ).strip()
    return choice_by_number[choice]


def _mask_secret(value: str, *, keep: int = 4) -> str:
    if len(value) <= keep * 2:
        return "*" * max(len(value), 3)
    return f"{value[:keep]}...{value[-keep:]}"


def _set_provider_api_key(settings, provider: str, api_key: str) -> None:
    if provider == "anthropic":
        settings.anthropic_api_key = api_key
    elif provider in {"openai", "openai-compatible"}:
        settings.openai_api_key = api_key
