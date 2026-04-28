from pathlib import Path
from typing import Iterable, Optional
import pandas as pd
from preprocessing.text_preprocessing import normalize_text


def load_welfake(path: Path, blocked_terms: Optional[Iterable[str]] = None) -> pd.DataFrame:
    df = pd.read_csv(path, usecols=["title", "text", "label"])
    df["title"] = normalize_text(df["title"], blocked_terms=blocked_terms)
    df["text"] = normalize_text(df["text"], blocked_terms=blocked_terms)
    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    df = df[df["label"].isin([0, 1])].copy()
    df["label"] = df["label"].astype(int)

    df["input_text"] = (df["title"] + " " + df["text"]).str.strip()
    df = df[df["input_text"].str.len() > 0].copy()
    return df[["input_text", "label"]].reset_index(drop=True)


def load_recovery(path: Path, blocked_terms: Optional[Iterable[str]] = None) -> pd.DataFrame:
    df = pd.read_csv(path, usecols=["title", "body_text", "reliability"])
    df["title"] = normalize_text(df["title"], blocked_terms=blocked_terms)
    df["body_text"] = normalize_text(df["body_text"], blocked_terms=blocked_terms)
    df["reliability"] = pd.to_numeric(df["reliability"], errors="coerce")
    df = df[df["reliability"].isin([0, 1])].copy()
    df["label"] = df["reliability"].astype(int)

    df["input_text"] = (df["title"] + " " + df["body_text"]).str.strip()
    df = df[df["input_text"].str.len() > 0].copy()
    return df[["input_text", "label"]].reset_index(drop=True)
