import ast
import json


def safe_parse(s: str) -> dict:
    """
    Parse a string that might be either JSON or a Python literal dict.
    Returns a Python dict.
    """
    if not s or not isinstance(s, str):
        raise ValueError("Input must be a non-empty string")

    try:
        # Try JSON first (since valid JSON should always work here).
        return json.loads(s)
    except json.JSONDecodeError:
        # If that fails, try Python literal (e.g., single quotes, True/False).
        return ast.literal_eval(s)
