"""External signals beyond StatsBomb: FIFA World Ranking trajectory and
pre-tournament form, weighted by opponent strength.

Directly addresses two methodological gaps flagged in review:
1. A goal against Qatar shouldn't count the same as a goal against England —
   `weighted_form_pre_tournament` discounts/boosts each result by the
   opponent's FIFA ranking points at the time.
2. In-tournament stats alone ignore the team's trajectory walking in — the
   FIFA ranking trend captures whether a team was rising or declining before
   the tournament started, which a pure box-score snapshot misses.

Sources (public, no auth, no scraping ToS issues):
- ranking_fifa_historical.csv: github.com/Dato-Futbol/fifa-ranking (1992-2024)
- international_results.csv: github.com/martj42/international_results
  (1872-2026, includes WC 2026 group-stage results played so far)
"""
import os

import numpy as np
import pandas as pd
import requests

EXTERNAL_DIR = os.path.join(os.path.dirname(__file__), "external")

SOURCE_URLS = {
    "fifa_ranking_historical.csv": (
        "https://raw.githubusercontent.com/Dato-Futbol/fifa-ranking/master/"
        "ranking_fifa_historical.csv"
    ),
    "international_results.csv": (
        "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
    ),
}


def _ensure_downloaded():
    os.makedirs(EXTERNAL_DIR, exist_ok=True)
    for filename, url in SOURCE_URLS.items():
        path = os.path.join(EXTERNAL_DIR, filename)
        if not os.path.exists(path):
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            with open(path, "wb") as f:
                f.write(resp.content)

# StatsBomb team name -> name used in each external dataset (they disagree
# with each other: rankings use "IR Iran"/"Korea Republic"/"USA", results
# uses "Iran"/"South Korea"/"United States").
RANKING_NAME_MAP = {
    "Iran": "IR Iran",
    "South Korea": "Korea Republic",
    "United States": "USA",
}
RESULTS_NAME_MAP: dict[str, str] = {}

TOURNAMENT_START = {2018: "2018-06-14", 2022: "2022-11-20"}
FORM_WINDOW_MATCHES = 15


def _ranking_name(team: str) -> str:
    return RANKING_NAME_MAP.get(team, team)


def _results_name(team: str) -> str:
    return RESULTS_NAME_MAP.get(team, team)


def _load_rankings() -> pd.DataFrame:
    _ensure_downloaded()
    df = pd.read_csv(os.path.join(EXTERNAL_DIR, "fifa_ranking_historical.csv"))
    df["date"] = pd.to_datetime(df["date"])
    return df


def _load_results() -> pd.DataFrame:
    _ensure_downloaded()
    df = pd.read_csv(os.path.join(EXTERNAL_DIR, "international_results.csv"))
    df["date"] = pd.to_datetime(df["date"])
    return df


def fifa_ranking_features(team: str, as_of: pd.Timestamp, rankings: pd.DataFrame) -> dict:
    """Ranking points just before `as_of`, and the 12-month trend leading up to it."""
    team_r = rankings[rankings["team"] == _ranking_name(team)].sort_values("date")
    prior = team_r[team_r["date"] <= as_of]
    if prior.empty:
        return {"fifa_points_pre": np.nan, "fifa_points_trend_12m": np.nan}
    current_points = prior.iloc[-1]["total_points"]
    year_ago = prior[prior["date"] <= as_of - pd.DateOffset(months=12)]
    trend = current_points - year_ago.iloc[-1]["total_points"] if not year_ago.empty else np.nan
    return {"fifa_points_pre": current_points, "fifa_points_trend_12m": trend}


def weighted_form_pre_tournament(
    team: str, as_of: pd.Timestamp, results: pd.DataFrame, rankings: pd.DataFrame
) -> dict:
    """Form over the last FORM_WINDOW_MATCHES matches before the tournament,
    each result weighted by the opponent's FIFA points at match time so beating
    a top-10 side counts more than beating a minnow."""
    ext_name = _results_name(team)
    mask = (
        ((results["home_team"] == ext_name) | (results["away_team"] == ext_name))
        & (results["date"] < as_of)
        & results["home_score"].notna()
    )
    matches = results[mask].sort_values("date").tail(FORM_WINDOW_MATCHES)
    if matches.empty:
        return {"weighted_form_score": np.nan, "weighted_goal_diff": np.nan}

    scores, goal_diffs = [], []
    for _, m in matches.iterrows():
        is_home = m["home_team"] == ext_name
        gf, ga = (m["home_score"], m["away_score"]) if is_home else (m["away_score"], m["home_score"])
        opponent = _ranking_name(m["away_team"] if is_home else m["home_team"])

        opp_rank = rankings[(rankings["team"] == opponent) & (rankings["date"] <= m["date"])]
        opp_points = opp_rank.sort_values("date").iloc[-1]["total_points"] if not opp_rank.empty else 1000.0
        weight = opp_points / 1000.0  # ~1.0 for a mid-table side, >1 for elite opposition

        result_points = 3 if gf > ga else (1 if gf == ga else 0)
        scores.append(result_points * weight)
        goal_diffs.append((gf - ga) * weight)

    return {
        "weighted_form_score": float(np.mean(scores)),
        "weighted_goal_diff": float(np.mean(goal_diffs)),
    }


def build_external_features(teams_years: list[tuple[str, int]]) -> pd.DataFrame:
    """teams_years: list of (team, year) for year in {2018, 2022}."""
    rankings = _load_rankings()
    results = _load_results()
    rows = []
    for team, year in teams_years:
        as_of = pd.Timestamp(TOURNAMENT_START[year])
        row = {"team": team, "year": year}
        row.update(fifa_ranking_features(team, as_of, rankings))
        row.update(weighted_form_pre_tournament(team, as_of, results, rankings))
        rows.append(row)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    tv = pd.read_csv("data/processed/team_vectors.csv")
    pairs = list(tv[["team", "year"]].itertuples(index=False, name=None))
    ext = build_external_features(pairs)
    ext.to_csv("data/processed/external_features.csv", index=False)
    print(ext.sort_values("weighted_form_score", ascending=False).head(10))
    print(f"\nMissing fifa_points_pre: {ext['fifa_points_pre'].isna().sum()} / {len(ext)}")
