"""Data loading and preprocessing utilities."""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd

from config import REFOREST_CONTEXTS


def clean_text(value: Any) -> str:
    """Normalize Portuguese textual descriptors for computational processing."""
    if pd.isna(value):
        return ""
    text = str(value).lower()
    text = re.sub(r"[^a-zà-ú0-9\s,;()-]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def convert_latitude(value: Any) -> float:
    """Extract the first latitude-like numeric value from a textual descriptor."""
    if pd.isna(value) or value == "":
        return np.nan
    try:
        return float(re.findall(r"-?\d+\.?\d*", str(value).lower().replace(",", ".").strip())[0])
    except (IndexError, ValueError):
        return np.nan


def climate_from_latitude(lat: float) -> str:
    """Infer a coarse Köppen-like climate class from latitude when climate is missing.

    This is a fallback proxy, not a precise climatic characterization.
    """
    if pd.isna(lat):
        return "Outro"
    if -10 <= lat <= 10:
        return "Af"
    if -30 <= lat <= -19:
        return "Cwa"
    if -27 <= lat <= -20:
        return "Cfa"
    if -35 <= lat <= -25:
        return "Cfb"
    return "Outro"


def extract_koeppen_codes(value: Any) -> list[str]:
    """Extract Köppen-Geiger codes from a text field."""
    if pd.isna(value) or value == "":
        return []
    codes = re.findall(r"[A-Z][a-z]{1,2}", str(value), re.IGNORECASE)
    return [code.upper() for code in codes]


def categorize_density(value: Any) -> str:
    """Convert numeric density-like values into broad abundance classes."""
    if pd.isna(value) or value == "":
        return ""
    try:
        density = float(value)
        if density > 0.70:
            return "alta"
        if density >= 0.50:
            return "media"
        return "baixa"
    except ValueError:
        return ""


def summarize_successional_group(value: Any) -> str:
    """Summarize textual successional descriptions into operational groups."""
    text = str(value).lower()
    groups: set[str] = set()

    if "pioneira" in text or "pion." in text:
        groups.add("Pioneira")
    if "secundaria inicial" in text or "secundaria in." in text or "secundaria, inicial" in text:
        groups.add("Secundária Inicial")
    if "secundaria tardia" in text or "secundaria final" in text or "tardia" in text:
        groups.add("Secundária Tardia")
    if "climax" in text or "clímace" in text or "clímax" in text or "clã­max" in text:
        groups.add("Clímax")
    if "inicial" in text and not groups:
        groups.add("Secundária Inicial")

    order = {"Pioneira": 1, "Secundária Inicial": 2, "Secundária Tardia": 3, "Clímax": 4}
    sorted_groups = sorted(groups, key=lambda x: order.get(x, 5))
    return ", ".join(sorted_groups) if sorted_groups else "Indefinido"


def extract_reforestation_contexts(value: Any) -> str:
    """Extract restoration contexts from textual descriptions using a controlled vocabulary."""
    text = clean_text(value)
    found = [ctx for ctx in REFOREST_CONTEXTS if ctx in text]
    return ", ".join(found) if found else "indefinido"


def extract_population_density(value: Any) -> float:
    """Extract population density values from textual descriptions.

    Returns the mean of all detected numeric abundance records.
    """
    if pd.isna(value) or value == "":
        return 0.0
    text = str(value).lower().replace("por hectare", "ha").replace("indivíduos", "ind")
    matches = re.findall(r"(\d+)\s+(?:arvore|arvores|ind|ha|individuos)", text)
    values = [int(match) for match in matches if match.isdigit()]
    return float(np.mean(values)) if values else 0.0


def clean_display_text(value: Any, max_len: int = 100) -> str:
    """Clean text for compact table display."""
    if pd.isna(value):
        return "N/A"
    text = str(value)
    try:
        text = text.encode("latin1").decode("utf-8", "ignore")
    except Exception:
        pass
    text = re.sub(r"[\r\n\t]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[: max_len - 3] + "..." if len(text) > max_len else text


def load_and_preprocess(csv_path: str) -> pd.DataFrame:
    """Load the curated species dataset and generate standardized modeling columns."""
    raw = pd.read_csv(csv_path, sep=";", encoding="latin1", dtype=str)
    raw.columns = [col.strip() for col in raw.columns]

    df = pd.DataFrame()
    df["nome"] = raw["nome"]
    df["nome_cientifico"] = raw.get("nome_cientifico", raw["nome"])
    df["latitude"] = raw["latitude"]
    df["grupo_sucessional"] = raw["grupo_sucessional"]
    df["densidade"] = raw["densidade"]
    df["tipos_climaticos"] = raw.get("tipos_climaticos (koeppen)", raw.get("tipos_climaticos", pd.NA))
    df["solo"] = raw["solo"]
    df["reflorestamento_original"] = raw.get(
        "reflorestamento_recuperacao ambiental:",
        raw.get("reflorestamento_recuperacao_ambiental", pd.NA),
    )
    df["dispersao"] = raw.get("dispersão de frutos e sementes", raw.get("dispersao", pd.NA))
    df["caracteristicas_silviculturais"] = raw.get(
        "características silviculturais", raw.get("caracteristicas_silviculturais", pd.NA)
    )
    df["crescimento_producao"] = raw.get("crescimento_produção", raw.get("crescimento", pd.NA))
    df["medicinal"] = raw.get("medicinal", pd.NA)
    df["oleo"] = raw.get("oleo", pd.NA)
    df["resina"] = raw.get("resina", pd.NA)

    df = df[df["nome"].notna() & (df["nome"] != "")]
    df["nome_cientifico"] = df["nome_cientifico"].replace(r"^\s*$", np.nan, regex=True)
    df = df[df["nome_cientifico"].notna()].reset_index(drop=True)

    text_cols = [
        "dispersao", "densidade", "solo", "caracteristicas_silviculturais",
        "crescimento_producao", "reflorestamento_original", "medicinal", "oleo", "resina",
    ]
    for col in text_cols:
        df[col] = df[col].apply(clean_text)

    df["latitude_num"] = df["latitude"].apply(convert_latitude)
    df["tipos_climaticos"] = df.apply(
        lambda row: climate_from_latitude(row["latitude_num"])
        if pd.isna(row["tipos_climaticos"]) or row["tipos_climaticos"] == ""
        else row["tipos_climaticos"],
        axis=1,
    )
    df["densidade"] = df["densidade"].apply(categorize_density)
    df["grupo_sucessional_resumido"] = df["grupo_sucessional"].apply(summarize_successional_group)
    df["contextos_reflorestamento"] = df["reflorestamento_original"].apply(extract_reforestation_contexts)
    df["densidade_populacional"] = df["reflorestamento_original"].apply(extract_population_density)
    return df
