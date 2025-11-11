import ast
import json


def safe_parse(s: str):
    """
    Parse a string that might be either JSON or a Python literal dict.
    Returns a Python dict.
    """
    try:
        # Try JSON first (since valid JSON should always work here).
        return json.loads(s)
    except json.JSONDecodeError:
        # If that fails, try Python literal (e.g., single quotes, True/False).
        return ast.literal_eval(s)