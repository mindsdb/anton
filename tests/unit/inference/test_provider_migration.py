"""Verify backward compat for moved provider modules."""


def test_openai_module_import_from_new_location():
    """OpenAI module can be imported from minds.inference.providers."""
    from minds.inference.providers import openai as openai_new

    assert openai_new is not None
    assert hasattr(openai_new, "proxy_openai")


def test_openai_module_import_from_old_location():
    """Old OpenAI imports still work (backward compat)."""
    from minds.agents.passthrough_agent import openai as openai_old

    assert openai_old is not None
    assert hasattr(openai_old, "proxy_openai")


def test_openai_modules_are_same():
    """Both import paths point to identical module."""
    from minds.agents.passthrough_agent import openai as openai_old
    from minds.inference.providers import openai as openai_new

    assert openai_new.proxy_openai is openai_old.proxy_openai


def test_anthropic_module_import_from_new_location():
    """Anthropic module can be imported from minds.inference.providers."""
    from minds.inference.providers import anthropic as anthropic_new

    assert anthropic_new is not None
    assert hasattr(anthropic_new, "proxy_anthropic")


def test_anthropic_module_import_from_old_location():
    """Old Anthropic imports still work (backward compat)."""
    from minds.agents.passthrough_agent import anthropic as anthropic_old

    assert anthropic_old is not None
    assert hasattr(anthropic_old, "proxy_anthropic")


def test_anthropic_modules_are_same():
    """Both import paths point to identical module."""
    from minds.agents.passthrough_agent import anthropic as anthropic_old
    from minds.inference.providers import anthropic as anthropic_new

    assert anthropic_new.proxy_anthropic is anthropic_old.proxy_anthropic


def test_gemini_module_import_from_new_location():
    """Gemini module can be imported from minds.inference.providers."""
    from minds.inference.providers import gemini as gemini_new

    assert gemini_new is not None
    assert hasattr(gemini_new, "proxy_gemini")


def test_gemini_module_import_from_old_location():
    """Old Gemini imports still work (backward compat)."""
    from minds.agents.passthrough_agent import gemini as gemini_old

    assert gemini_old is not None
    assert hasattr(gemini_old, "proxy_gemini")


def test_gemini_modules_are_same():
    """Both import paths point to identical module."""
    from minds.agents.passthrough_agent import gemini as gemini_old
    from minds.inference.providers import gemini as gemini_new

    assert gemini_new.proxy_gemini is gemini_old.proxy_gemini
