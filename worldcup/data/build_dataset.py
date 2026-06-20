"""Orchestrates Task 1: builds team features, player features, and the
combined team vector = [team_features + mean(top-5 player_features)] for
each historical tournament (2018, 2022). Run as a script to regenerate
everything under data/processed/.
"""
import os

import pandas as pd

from data.external_signals import build_external_features
from data.player_features import build_player_features
from data.team_features import build_team_features

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "processed")

TEAM_FEATURE_COLS = [
    "avg_possession_pct",
    "avg_xg_for",
    "avg_xg_against",
    "avg_shots_on_target",
    "avg_pass_completion_pct",
    "avg_ppda",
    "avg_high_pressures",
]
PLAYER_FEATURE_COLS = [
    "goals_plus_assists",
    "key_passes_per90",
    "xg_plus_xa",
    "duel_win_pct",
]


def build_team_vectors(years=(2018, 2022)) -> pd.DataFrame:
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    team_frames, player_frames = [], []

    for year in years:
        team_df = build_team_features(year)
        player_df = build_player_features(year)
        team_df.to_csv(f"{PROCESSED_DIR}/team_features_{year}.csv", index=False)
        player_df.to_csv(f"{PROCESSED_DIR}/player_features_{year}.csv", index=False)
        team_frames.append(team_df)
        player_frames.append(player_df)

    teams = pd.concat(team_frames, ignore_index=True)
    players = pd.concat(player_frames, ignore_index=True)

    star_avg = (
        players.groupby(["year", "team"])[PLAYER_FEATURE_COLS]
        .mean()
        .reset_index()
        .rename(columns={c: f"star_{c}" for c in PLAYER_FEATURE_COLS})
    )

    vectors = teams.merge(star_avg, on=["year", "team"], how="left")

    pairs = list(vectors[["team", "year"]].itertuples(index=False, name=None))
    external = build_external_features(pairs)
    external.to_csv(f"{PROCESSED_DIR}/external_features.csv", index=False)
    vectors = vectors.merge(external, on=["year", "team"], how="left")

    vectors.to_csv(f"{PROCESSED_DIR}/team_vectors.csv", index=False)
    return vectors


if __name__ == "__main__":
    df = build_team_vectors()
    print(f"Built {len(df)} team vectors -> data/processed/team_vectors.csv")
    print(df.head())
