"""Cosine-similarity engine: for each team in a target tournament, find its
most similar team(s) in a source tournament, based on a standardized feature
vector.

`round_reached` (the eventual prediction target) and tournament-volume
features (`matches_played`, raw `goals_for`/`goals_against`, which mechanically
correlate with how far a team advanced) are excluded from the similarity
vector to avoid leaking the very thing we're trying to validate.
"""
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

LEAKAGE_COLS = {"matches_played", "goals_for", "goals_against", "round_reached", "team", "year"}


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in LEAKAGE_COLS]


def standardized_vectors(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    return StandardScaler().fit_transform(df[cols].fillna(df[cols].mean()))


def most_similar(
    target_df: pd.DataFrame,
    source_df: pd.DataFrame,
    cols: list[str],
    top_n: int = 1,
) -> pd.DataFrame:
    """For each row in target_df, return the top_n most similar rows in source_df.

    Standardization is fit jointly across target+source so both are on the
    same scale, then similarity is computed target-vs-source only.
    """
    combined = pd.concat([target_df, source_df], ignore_index=True)
    vectors = standardized_vectors(combined, cols)
    target_vecs = vectors[: len(target_df)]
    source_vecs = vectors[len(target_df) :]

    sims = cosine_similarity(target_vecs, source_vecs)

    rows = []
    for i, target_team in enumerate(target_df["team"]):
        order = np.argsort(-sims[i])[:top_n]
        for rank, j in enumerate(order, start=1):
            rows.append(
                {
                    "team": target_team,
                    "match_rank": rank,
                    "twin_team": source_df.iloc[j]["team"],
                    "twin_year": source_df.iloc[j]["year"],
                    "similarity_pct": sims[i, j] * 100,
                    "twin_round_reached": source_df.iloc[j]["round_reached"],
                }
            )
    return pd.DataFrame(rows)
