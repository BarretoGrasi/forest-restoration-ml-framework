"""Operational planting plan generation from ranked species."""

from __future__ import annotations

import numpy as np
import pandas as pd

STAGE_MAP = {
    "Pioneira": "Pioneira",
    "Secundária Inicial": "Sec. Inicial",
    "Secundária Tardia": "Sec. Tardia",
    "Clímax": "Clímax",
}

DEFAULT_STAGE_PROPORTIONS = {
    "Pioneira": 0.25,
    "Sec. Inicial": 0.40,
    "Sec. Tardia": 0.25,
    "Clímax": 0.10,
}


def generate_planting_plan(
    ranked_df: pd.DataFrame,
    top_n: int = 10,
    area_ha: float = 1.0,
    density_per_ha: int = 2500,
    min_seedlings_per_species: int = 10,
    proportions: dict | None = None,
) -> pd.DataFrame:
    """Allocate seedlings by successional group and hybrid score."""
    proportions = proportions or DEFAULT_STAGE_PROPORTIONS
    plan = ranked_df.head(top_n).copy()
    total_seedlings = int(area_ha * density_per_ha)
    remaining = total_seedlings - len(plan) * min_seedlings_per_species

    plan["Qtd. Mudas"] = min_seedlings_per_species
    for stage in STAGE_MAP.values():
        plan[f"Contrib. {stage}"] = 0

    def parse_stages(text: str) -> list[str]:
        return [STAGE_MAP[s.strip()] for s in str(text).replace("\n", "").split(",") if s.strip() in STAGE_MAP]

    plan["estagios_chaves"] = plan["grupo_sucessional_resumido"].apply(parse_stages)

    for stage_key, prop in proportions.items():
        volume_stage = prop * remaining
        species_stage = plan[plan["estagios_chaves"].apply(lambda stages: stage_key in stages)]
        total_score = species_stage["pontuacao_hibrida"].sum()
        if total_score > 0:
            allocation = ((species_stage["pontuacao_hibrida"] / total_score) * volume_stage).round().astype(int)
            plan.loc[species_stage.index, f"Contrib. {stage_key}"] = allocation
            plan.loc[species_stage.index, "Qtd. Mudas"] += allocation

    difference = total_seedlings - plan["Qtd. Mudas"].sum()
    if difference != 0:
        best_idx = plan["pontuacao_hibrida"].idxmax()
        plan.loc[best_idx, "Qtd. Mudas"] += difference

    spacing = np.sqrt(10000 / density_per_ha)
    plan["Espaçamento"] = f"{spacing:.1f}m x {spacing:.1f}m"
    return plan.drop(columns=["estagios_chaves"])
