"""Machine learning training and ranking pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from config import ML_TEXT_COLUMNS, RANDOM_STATE
from scoring import ecological_score


def build_feature_matrix(df: pd.DataFrame, max_features: int = 800, ngram_range: tuple = (1, 2)):
    """Create TF-IDF + numeric feature matrix."""
    df = df.copy()
    df["features_texto"] = df[ML_TEXT_COLUMNS].astype(str).agg(" ".join, axis=1)
    vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)
    x_text = vectorizer.fit_transform(df["features_texto"]).toarray()
    x_num = df[["densidade_populacional"]].values
    x = np.hstack((x_text, x_num))
    return x, vectorizer, df


def train_surrogate_model(df: pd.DataFrame, base_profile: dict, weights: dict):
    """Train Random Forest surrogate model to emulate expert-derived ecological scores."""
    df = df.copy()
    df["pontuacao_target"] = df.apply(lambda row: ecological_score(row, base_profile, weights)[0], axis=1)
    y = df["pontuacao_target"].values
    x, vectorizer, df = build_feature_matrix(df)

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=RANDOM_STATE
    )
    model = RandomForestRegressor(n_estimators=300, max_depth=12, random_state=RANDOM_STATE)
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)
    metrics = {
        "r2": r2_score(y_test, y_pred),
        "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
    }
    return model, vectorizer, df, metrics


def rank_species(df: pd.DataFrame, model, vectorizer, user_profile: dict, weights: dict, alpha: float = 0.5) -> pd.DataFrame:
    """Generate hybrid ranking by combining expert score and surrogate ML score."""
    df = df.copy()
    df["pontuacao_ecologica_final"] = df.apply(lambda row: ecological_score(row, user_profile, weights)[0], axis=1)

    x_text = vectorizer.transform(df["features_texto"]).toarray()
    x_num = df[["densidade_populacional"]].values
    x_all = np.hstack((x_text, x_num))
    df["pontuacao_ml_prevista"] = model.predict(x_all)
    df["pontuacao_hibrida"] = alpha * df["pontuacao_ecologica_final"] + (1 - alpha) * df["pontuacao_ml_prevista"]
    return df.sort_values("pontuacao_hibrida", ascending=False)
