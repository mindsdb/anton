from datetime import datetime

from minds.agents.helpers.chart_generation_instructions import CHART_GENERATION_INSTRUCTIONS
from minds.agents.llm import get_llm_config
from minds.common.settings.app_settings import Agent, Settings, get_app_settings
from minds.model.mind import Mind

app_settings = get_app_settings()


def model_for(mind: Mind):
    return get_llm_config(mind.provider, mind.model_name)


def mind_layer(mind: Mind) -> str:
    return mind.parameters.get("system_prompt") or mind.parameters.get("prompt_template") or ""


def charting_layer() -> str:
    return CHART_GENERATION_INSTRUCTIONS


def current_date_time_layer() -> str:
    return f"The current date and time is {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"


def is_native_query_mode_enabled(mind: Mind, settings: Settings) -> bool:
    engines = set([mind_datasource.datasource.engine for mind_datasource in mind.mind_datasources])
    return (
        mind.parameters.get(
            "use_native_query_mode",
            getattr(settings, "use_native_query_mode", False),
        )
        and len(engines) == 1
        and engines.pop() in getattr(settings, "native_query_mode_supported_engines", [])
    )


def get_agent(mind: Mind) -> Agent:
    """Get the agent enum for the mind."""
    raw_agent_name = mind.parameters.get("agent_name") if mind.parameters else None
    try:
        agent = Agent(raw_agent_name) if raw_agent_name else app_settings.agents.default_agent
    except ValueError as e:
        allowed = ", ".join(a.value for a in Agent)
        raise ValueError(f"Invalid agent name: {raw_agent_name!r}. Allowed: {allowed}") from e
    return agent


def is_anton_agent(mind: Mind) -> bool:
    """Check if the mind is using the Anton agent."""
    return get_agent(mind) == Agent.ANTON
