"""Loads StatsBomb open-data for World Cup tournaments, with on-disk caching.

Confirmed available via open data (no credentials):
- competitions / matches (with competition_stage for round mapping)
- events (shot_statsbomb_xg, pass_outcome, possession_team, Pressure, Duel)
- lineups (player positions with from/to timestamps -> minutes played)

Aggregated stats endpoints (team_season_stats, player_season_stats) require
paid credentials and are NOT used; everything is derived from raw events.
"""
import ast
import os

import pandas as pd
from statsbombpy import sb

RAW_DIR = os.path.join(os.path.dirname(__file__), "raw")
os.makedirs(RAW_DIR, exist_ok=True)

WORLD_CUP_COMPETITION_ID = 43
SEASON_IDS = {2018: 3, 2022: 106}


def _cache_path(name: str) -> str:
    return os.path.join(RAW_DIR, name)


def get_matches(year: int) -> pd.DataFrame:
    path = _cache_path(f"matches_{year}.parquet")
    if os.path.exists(path):
        return pd.read_parquet(path)
    df = sb.matches(competition_id=WORLD_CUP_COMPETITION_ID, season_id=SEASON_IDS[year])
    df.to_parquet(path)
    return df


def get_events(match_id: int) -> pd.DataFrame:
    path = _cache_path(f"events_{match_id}.parquet")
    if os.path.exists(path):
        return pd.read_parquet(path)
    df = sb.events(match_id=match_id)
    # drop nested freeze-frame/object columns that parquet can't serialize cleanly
    obj_cols = [c for c in df.columns if df[c].dtype == "object"]
    for c in obj_cols:
        df[c] = df[c].apply(lambda v: v if not isinstance(v, (list, dict)) else str(v))
    df.to_parquet(path)
    return df


def get_lineups(match_id: int) -> dict[str, pd.DataFrame]:
    path = _cache_path(f"lineups_{match_id}.parquet")
    if os.path.exists(path):
        df = pd.read_parquet(path)
        df["positions"] = df["positions"].apply(ast.literal_eval)
        return {team: g.drop(columns=["team"]).reset_index(drop=True) for team, g in df.groupby("team")}
    lineups = sb.lineups(match_id=match_id)
    frames = []
    for team, df in lineups.items():
        df = df.copy()
        df["team"] = team
        df["cards"] = df["cards"].apply(str)
        frames.append(df.assign(positions=df["positions"].apply(str)))
    combined = pd.concat(frames, ignore_index=True)
    combined.to_parquet(path)
    return lineups


def get_all_events_for_year(year: int) -> pd.DataFrame:
    """Concatenate events for every match of a tournament, tagged with match_id/team context."""
    matches = get_matches(year)
    all_events = []
    for match_id in matches["match_id"]:
        ev = get_events(match_id)
        ev["match_id"] = match_id
        all_events.append(ev)
    return pd.concat(all_events, ignore_index=True)
