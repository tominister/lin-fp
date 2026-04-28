import re
from typing import Iterable, Optional

import pandas as pd


def _compile_block_pattern(blocked_terms: Iterable[str]) -> Optional[str]:
    cleaned = [re.escape(term.strip().lower()) for term in blocked_terms if term and term.strip()]
    if not cleaned:
        return None
    phrase_pattern = "|".join(term.replace(r"\ ", r"\s+") for term in cleaned)
    return rf"\b(?:{phrase_pattern})\b"


def normalize_text(series: pd.Series, blocked_terms: Optional[Iterable[str]] = None) -> pd.Series:
    normalized = (
        series.fillna("")
        .astype(str)
        .str.lower()
        .str.replace(r"https?://\S+|www\.\S+", " ", regex=True)
        .str.replace(r"[^a-z0-9\s]", " ", regex=True)
    )

    block_pattern = _compile_block_pattern(blocked_terms or [])
    if block_pattern:
        normalized = normalized.str.replace(block_pattern, " ", regex=True)

    return normalized.str.replace(r"\s+", " ", regex=True).str.strip()
