"""Run complete forest restoration ML workflow.

Example:
    python src/main.py --data data/dataset_floresta2.csv --output-dir outputs
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from config import ECOLOGICAL_WEIGHTS
from modeling import rank_species, train_surrogate_model
from planting_plan import generate_planting_plan
from preprocessing import clean_display_text, load_and_preprocess
from scoring import build_user_profile


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Forest restoration ML framework")
    parser.add_argument("--data", required=True, help="Path to curated CSV dataset")
    parser.add_argument("--output-dir", default="outputs", help="Directory for output CSV files")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_and_preprocess(args.data)

    base_profile = build_user_profile({
        "latitude": -34.0,
        "solo": "franco-argilosa, fertil, umido",
        "caracteristicas_silviculturais": "sombreamento",
        "crescimento_producao": "moderado",
        "densidade": "0.15",
        "reflorestamento": "matas ciliar, areas degradadas",
    })

    user_profile = build_user_profile({
        "latitude": -34.25,
        "solo": "umidos pedregosos",
        "caracteristicas_silviculturais": "sombreamento",
        "crescimento_producao": "lento",
        "densidade": "0.15",
        "reflorestamento": "áreas degradadas",
    })

    model, vectorizer, df_features, metrics = train_surrogate_model(df, base_profile, ECOLOGICAL_WEIGHTS)
    ranked = rank_species(df_features, model, vectorizer, user_profile, ECOLOGICAL_WEIGHTS, alpha=0.5)
    
    # Alpha ablation analysis
    alpha_values = [0.0, 0.25, 0.5, 0.75, 1.0]

    for alpha in alpha_values:
        ranked_alpha = rank_species(
            df_features,
            model,
            vectorizer,
            user_profile,
            ECOLOGICAL_WEIGHTS,
            alpha=alpha,
        )

        top_alpha = ranked_alpha.head(10).copy()
        top_alpha["alpha"] = alpha

        selected_columns = [
            "alpha",
            "nome",
            "nome_cientifico",
            "grupo_sucessional_resumido",
            "pontuacao_ecologica_final",
            "pontuacao_ml_prevista",
            "pontuacao_hibrida",
        ]

        top_alpha[selected_columns].to_csv(
            output_dir / f"top_10_alpha_{str(alpha).replace('.', '_')}.csv",
            index=False,
            sep=";",
            encoding="utf-8",
        )
    top_species = ranked.head(10).copy()
    top_species["Uso Primário Registrado"] = top_species["reflorestamento_original"].apply(clean_display_text)
    top_species["Grupo Sucessional"] = top_species["grupo_sucessional_resumido"]

    columns = ["nome", "nome_cientifico", "Grupo Sucessional", "pontuacao_hibrida", "Uso Primário Registrado"]
    top_species[columns].to_csv(output_dir / "top_10_species.csv", index=False, sep=";", encoding="utf-8")

    plan = generate_planting_plan(ranked, top_n=10)
    plan.to_csv(output_dir / "planting_plan.csv", index=False, sep=";", encoding="utf-8")

    pd.DataFrame([metrics]).to_csv(output_dir / "model_metrics.csv", index=False, sep=";", encoding="utf-8")

    print("Workflow completed.")
    print(f"R²={metrics['r2']:.3f}; RMSE={metrics['rmse']:.3f}")
    print(f"Outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
