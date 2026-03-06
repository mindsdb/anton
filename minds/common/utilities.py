import ast
import json

import numpy as np
import pandas as pd


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


def format_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Format numeric columns to prevent scientific notation.

    Args:
        df: The DataFrame to format.

    Returns:
        The formatted DataFrame.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    for col in numeric_cols:
        if pd.api.types.is_float_dtype(df[col]):
            df[col] = df[col].map(lambda x: f"{x:,.1f}" if pd.notnull(x) else x)
        else:
            df[col] = df[col].map(lambda x: f"{x:,}" if pd.notnull(x) else x)

    return df
