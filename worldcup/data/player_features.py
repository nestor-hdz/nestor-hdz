"""Per-player, per-tournament feature extraction for each team's top-5 players
by minutes played (the "star player" proxy requested in the spec).

Assists are derived from `pass_assisted_shot_id`: a pass assists a goal when
the shot it created has `shot_outcome == 'Goal'`. xA proxies similarly: the
xG of every shot a player's pass created, regardless of outcome.
"""
import numpy as np
import pandas as pd

from data.minutes import get_period_end_minutes, player_minutes
from data.statsbomb_loader import get_events, get_lineups, get_matches

DUEL_WON_OUTCOMES = {"Won", "Success In Play", "Success Out"}
TOP_N_PER_TEAM = 5


def _match_player_stats(events: pd.DataFrame) -> pd.DataFrame:
    shots_by_id = events.set_index("id") if "id" in events.columns else None

    rows = []
    for team in events["team"].dropna().unique():
        team_events = events[events["team"] == team]
        for player_id, player_events in team_events.groupby("player_id"):
            player_name = player_events["player"].iloc[0]

            goals = (
                (player_events["type"] == "Shot") & (player_events["shot_outcome"] == "Goal")
            ).sum()

            passes = player_events[player_events["type"] == "Pass"]
            key_passes = passes["pass_shot_assist"].fillna(False).astype(bool).sum()

            assists = 0
            xa = 0.0
            if shots_by_id is not None:
                assisted_ids = passes["pass_assisted_shot_id"].dropna()
                for shot_id in assisted_ids:
                    if shot_id in shots_by_id.index:
                        shot = shots_by_id.loc[shot_id]
                        xa += shot.get("shot_statsbomb_xg", 0) or 0
                        if shot.get("shot_outcome") == "Goal":
                            assists += 1

            xg = player_events.loc[player_events["type"] == "Shot", "shot_statsbomb_xg"].sum()

            duels = player_events[player_events["type"] == "Duel"]
            duel_outcomes = duels["duel_outcome"].dropna()
            duels_won = duel_outcomes.isin(DUEL_WON_OUTCOMES).sum()

            rows.append(
                {
                    "team": team,
                    "player_id": player_id,
                    "player_name": player_name,
                    "goals": goals,
                    "assists": assists,
                    "key_passes": key_passes,
                    "xg": xg,
                    "xa": xa,
                    "duels_total": len(duel_outcomes),
                    "duels_won": duels_won,
                }
            )
    return pd.DataFrame(rows)


def build_player_features(year: int) -> pd.DataFrame:
    matches = get_matches(year)
    all_match_stats = []
    all_minutes = []

    for _, match in matches.iterrows():
        match_id = match["match_id"]
        events = get_events(match_id)
        events["team"] = events["team"].astype(str)

        stats = _match_player_stats(events)
        stats["match_id"] = match_id
        all_match_stats.append(stats)

        period_end = get_period_end_minutes(events)
        lineups = get_lineups(match_id)
        for team, lineup_df in lineups.items():
            mins = player_minutes(lineup_df, period_end).reset_index()
            mins["team"] = team
            mins["match_id"] = match_id
            all_minutes.append(mins)

    match_stats = pd.concat(all_match_stats, ignore_index=True)
    minutes_df = pd.concat(all_minutes, ignore_index=True)

    totals = (
        match_stats.groupby(["team", "player_id", "player_name"])
        .agg(
            goals=("goals", "sum"),
            assists=("assists", "sum"),
            key_passes=("key_passes", "sum"),
            xg=("xg", "sum"),
            xa=("xa", "sum"),
            duels_total=("duels_total", "sum"),
            duels_won=("duels_won", "sum"),
        )
        .reset_index()
    )
    minutes_totals = minutes_df.groupby(["team", "player_id"])["minutes"].sum().reset_index()
    merged = totals.merge(minutes_totals, on=["team", "player_id"], how="left")
    merged["minutes"] = merged["minutes"].fillna(0)

    merged["key_passes_per90"] = np.where(
        merged["minutes"] > 0, merged["key_passes"] / merged["minutes"] * 90, 0
    )
    merged["duel_win_pct"] = np.where(
        merged["duels_total"] > 0, merged["duels_won"] / merged["duels_total"] * 100, np.nan
    )
    merged["goals_plus_assists"] = merged["goals"] + merged["assists"]
    merged["xg_plus_xa"] = merged["xg"] + merged["xa"]
    merged["year"] = year

    merged = merged.sort_values(["team", "minutes"], ascending=[True, False])
    top5 = merged.groupby("team").head(TOP_N_PER_TEAM).reset_index(drop=True)
    return top5[
        [
            "year",
            "team",
            "player_id",
            "player_name",
            "minutes",
            "goals",
            "assists",
            "goals_plus_assists",
            "key_passes_per90",
            "xg",
            "xa",
            "xg_plus_xa",
            "duel_win_pct",
        ]
    ]


if __name__ == "__main__":
    for yr in (2018, 2022):
        df = build_player_features(yr)
        out_path = f"data/processed/player_features_{yr}.csv"
        df.to_csv(out_path, index=False)
        print(f"{yr}: {len(df)} player rows -> {out_path}")
