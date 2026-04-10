from pathlib import Path


_BOOT_SCRIPT_PATH = Path(__file__).parent / "scratchpad_boot.py"
# Package root for `import anton...` in scratchpad runtimes.
_ANTON_PKG_PATH = Path(__file__).parents[2]
_LLM_PROVIDER_PKG_PATH = Path(__file__).parent.parent / "llm"