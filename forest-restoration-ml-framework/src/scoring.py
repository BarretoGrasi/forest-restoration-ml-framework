"""Ecological compatibility scoring functions."""

from __future__ import annotations

import pandas as pd

from config import SOIL_TERMS
from preprocessing import clean_text, categorize_density, climate_from_latitude, extract_koeppen_codes, extract_population_density, extract_reforestation_contexts


def ecological_score(row: pd.Series, profile: dict, weights: dict) -> tuple[float, dict]:
    """Compute expert-derived ecological compatibility score for one species."""
    lat_species = row["latitude_num"]
    lat_profile = profile["latitude"]
    compat_lat = 1 - (abs(lat_species - lat_profile) / 40) if not pd.isna(lat_species) else 0.5
    compat_lat = max(0, min(1, compat_lat))

    species_climates = extract_koeppen_codes(row["tipos_climaticos"])
    profile_climate = profile["tipos_climaticos"].upper()
    if profile_climate in species_climates:
        compat_climate = 1.0
    elif not species_climates or profile_climate == "OUTRO":
        compat_climate = 0.5
    else:
        compat_climate = 0.2

    profile_soil = profile["solo"]
    species_soil = row["solo"]
    profile_terms = [term for term in SOIL_TERMS if term in profile_soil]
    compat_soil = 0.5
    if profile_terms:
        compat_soil = 1.0 if any(term in species_soil for term in profile_terms) else 0.7

    group_text = row["grupo_sucessional_resumido"].lower()
    compat_succession = 1.0 if any(
        group in group_text for group in ["pioneira", "secundaria inicial", "secundaria tardia", "climax"]
    ) else 0.6

    species_contexts = row["contextos_reflorestamento"]
    profile_contexts = profile["contextos_reflorestamento"]
    compat_reforestation = 0.6
    if profile_contexts != "indefinido":
        profile_list = [ctx.strip() for ctx in profile_contexts.split(", ") if ctx.strip()]
        if any(ctx in species_contexts for ctx in profile_list):
            compat_reforestation = 1.0

    profile_growth = profile.get("crescimento_producao", "").lower()
    species_growth = row["crescimento_producao"].lower()
    compat_growth = 0.5
    if "rapido" in profile_growth:
        compat_growth = 1.0 if "rapido" in species_growth else 0.7 if "moderado" in species_growth else 0.3
    elif "lento" in profile_growth:
        compat_growth = 1.0 if "lento" in species_growth else 0.7 if "moderado" in species_growth else 0.3
    else:
        if "moderado" in species_growth:
            compat_growth = 1.0
        elif "rapido" in species_growth or "lento" in species_growth:
            compat_growth = 0.7

    score = (
        weights["latitude"] * compat_lat
        + weights["clima"] * compat_climate
        + weights["solo"] * compat_soil
        + weights["sucessao"] * compat_succession
        + weights["reflorestamento"] * compat_reforestation
        + weights.get("crescimento", 0.0) * compat_growth
    )
    details = {
        "latitude": compat_lat,
        "clima": compat_climate,
        "solo": compat_soil,
        "sucessao": compat_succession,
        "reflorestamento": compat_reforestation,
        "crescimento": compat_growth,
    }
    return score, details


def build_user_profile(raw_profile: dict) -> dict:
    """Clean and complete user-defined restoration profile."""
    profile = dict(raw_profile)
    profile["solo"] = clean_text(profile["solo"])
    profile["densidade"] = categorize_density(str(profile.get("densidade", "")))
    profile["tipos_climaticos"] = climate_from_latitude(profile["latitude"])
    profile["contextos_reflorestamento"] = extract_reforestation_contexts(profile.get("reflorestamento", ""))
    profile["densidade_populacional"] = extract_population_density(profile.get("reflorestamento", ""))
    profile["crescimento_producao"] = clean_text(profile.get("crescimento_producao", ""))
    return profile
