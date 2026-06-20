"""Per-team, per-tournament feature extraction from StatsBomb events.

PPDA (Passes Per Defensive Action) proxy: opponent passes completed in their
own defensive 60% of the pitch, divided by the team's defensive actions
(Pressure, Duel, Interception, Foul Committed) in that same zone. Lower PPDA
= more aggressive high press. StatsBomb pitch coordinates are not normalized
to a fixed attacking direction, so direction is inferred per team/period from
the mean x-location of that team's shots (attacks converge toward the
opponent's goal), then all locations are flipped so each team always
"attacks" toward x=120.
"""
import ast

import numpy as np
import pandas as pd

from data.statsbomb_loader import get_events, get_matches

ON_TARGET_OUTCOMES = {"Goal", "Saved", "Saved to Post", "Post"}
DEFENSIVE_ACTION_TYPES = {"Pressure", "Duel", "Interception", "Foul Committed"}

ROUND_ORDER = [
    "Group Stage",
    "Round of 16",
    "Quarter-finals",
    "Semi-finals",
    "Final",
]
ROUND_TO_CODE = {name: i for i, name in enumerate(ROUND_ORDER)}


def _parse_loc(v):
    if isinstance(v, str):
        try:
            v = ast.literal_eval(v)
        except (ValueError, SyntaxError):
            return None
    if isinstance(v, (list, tuple)) and len(v) >= 1:
        return v[0]
    return None


def _attack_direction_by_team_period(events: pd.DataFrame) -> dict:
    """Returns {(team, period): +1 if attacking toward x=120 else -1}."""
    shots = events[events["type"] == "Shot"].copy()
    shots["x"] = shots["location"].apply(_parse_loc)
    directions = {}
    for (team, period), g in shots.groupby(["team", "period"]):
        mean_x = g["x"].dropna().mean()
        directions[(team, period)] = 1 if (pd.isna(mean_x) or mean_x >= 60) else -1
    return directions


def _normalized_x(events: pd.DataFrame, directions: dict) -> pd.Series:
    x = events["location"].apply(_parse_loc)
    dirs = events.apply(lambda r: directions.get((r["team"], r["period"]), 1), axis=1)
    return np.where(dirs == 1, x, 120 - x)


def _team_round_reached(matches: pd.DataFrame) -> dict:
    """Deepest stage played per team; champion gets a code one above Final."""
    reached = {}
    for _, m in matches.iterrows():
        stage = m["competition_stage"]
        if stage not in ROUND_TO_CODE:
            continue  # skip 3rd Place Final, doesn't add depth beyond Semis
        code = ROUND_TO_CODE[stage]
        for team in (m["home_team"], m["away_team"]):
            reached[team] = max(reached.get(team, 0), code)
    final = matches[matches["competition_stage"] == "Final"]
    if not final.empty:
        f = final.iloc[0]
        if f["home_score"] != f["away_score"]:
            winner = f["home_team"] if f["home_score"] > f["away_score"] else f["away_team"]
        else:
            # decided on penalties: count shootout (period 5) goals
            shootout = get_events(f["match_id"])
            shootout = shootout[(shootout["period"] == 5) & (shootout["type"] == "Shot")]
            pens = shootout[shootout["shot_outcome"] == "Goal"]["team"].value_counts()
            winner = pens.idxmax() if not pens.empty else f["home_team"]
        reached[winner] = ROUND_TO_CODE["Final"] + 1  # champion
    return reached


def build_team_features(year: int) -> pd.DataFrame:
    matches = get_matches(year)
    round_reached = _team_round_reached(matches)

    rows = []
    for _, match in matches.iterrows():
        match_id = match["match_id"]
        events = get_events(match_id)
        events["team"] = events["team"].astype(str)
        directions = _attack_direction_by_team_period(events)
        events = events.assign(x_norm=_normalized_x(events, directions))

        for team in (match["home_team"], match["away_team"]):
            opp = match["away_team"] if team == match["home_team"] else match["home_team"]
            team_events = events[events["team"] == team]
            opp_events = events[events["team"] == opp]

            shots = team_events[team_events["type"] == "Shot"]
            xg_for = shots["shot_statsbomb_xg"].sum()
            on_target = shots["shot_outcome"].isin(ON_TARGET_OUTCOMES).sum()

            opp_shots = opp_events[opp_events["type"] == "Shot"]
            xg_against = opp_shots["shot_statsbomb_xg"].sum()

            passes = team_events[team_events["type"] == "Pass"]
            completed_passes = passes["pass_outcome"].isna().sum()
            pass_pct = completed_passes / len(passes) if len(passes) else np.nan

            possession_pct = (
                events[events["possession_team"] == team]["possession"].nunique()
                / events["possession"].nunique()
                * 100
                if events["possession"].nunique()
                else np.nan
            )

            # PPDA proxy: opponent passes completed in their own defensive 60%
            # (x_norm < 80, attacking toward 120) vs team's defensive actions there.
            opp_passes_def_zone = opp_events[
                (opp_events["type"] == "Pass")
                & opp_events["pass_outcome"].isna()
                & (opp_events["x_norm"] < 80)
            ]
            team_def_actions_zone = team_events[
                team_events["type"].isin(DEFENSIVE_ACTION_TYPES) & (team_events["x_norm"] < 80)
            ]
            ppda = (
                len(opp_passes_def_zone) / len(team_def_actions_zone)
                if len(team_def_actions_zone)
                else np.nan
            )
            high_pressures = team_events[
                (team_events["type"] == "Pressure") & (team_events["x_norm"] >= 80)
            ].shape[0]

            goals_for = match["home_score"] if team == match["home_team"] else match["away_score"]
            goals_against = match["away_score"] if team == match["home_team"] else match["home_score"]

            rows.append(
                {
                    "year": year,
                    "match_id": match_id,
                    "team": team,
                    "opponent": opp,
                    "goals_for": goals_for,
                    "goals_against": goals_against,
                    "possession_pct": possession_pct,
                    "xg_for": xg_for,
                    "xg_against": xg_against,
                    "shots_on_target": on_target,
                    "pass_completion_pct": pass_pct * 100 if pd.notna(pass_pct) else np.nan,
                    "ppda": ppda,
                    "high_pressures": high_pressures,
                }
            )

    match_level = pd.DataFrame(rows)
    agg = (
        match_level.groupby(["year", "team"])
        .agg(
            matches_played=("match_id", "count"),
            goals_for=("goals_for", "sum"),
            goals_against=("goals_against", "sum"),
            avg_possession_pct=("possession_pct", "mean"),
            avg_xg_for=("xg_for", "mean"),
            avg_xg_against=("xg_against", "mean"),
            avg_shots_on_target=("shots_on_target", "mean"),
            avg_pass_completion_pct=("pass_completion_pct", "mean"),
            avg_ppda=("ppda", "mean"),
            avg_high_pressures=("high_pressures", "mean"),
        )
        .reset_index()
    )
    agg["round_reached"] = agg["team"].map(round_reached)
    return agg


if __name__ == "__main__":
    for yr in (2018, 2022):
        df = build_team_features(yr)
        out_path = f"data/processed/team_features_{yr}.csv"
        df.to_csv(out_path, index=False)
        print(f"{yr}: {len(df)} teams -> {out_path}")
