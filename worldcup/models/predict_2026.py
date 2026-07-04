"""World Cup 2026 comparisons — NOT a predictor.

This module deliberately does not output a predicted round, a round-by-round
probability, or a championship probability for 2026 teams. Early in this
project a prior pure-stats approach presented a similarity score as a
confident forecast ("Argentina will be champion") and that was wrong and
misleading. The agreed fix (see DECISIONS.md) is to keep the output strictly
comparative: "2026 team X resembles historical team Y by Z%, based on these
signals" - what that historical team went on to do is shown as context, not
as a forecast for the 2026 team.

Why the feature set is reduced for 2026: StatsBomb has no event-level data
for World Cup 2026 (it hasn't been played in full, and FBref/Transfermarkt/
ESPN are blocked by this sandbox's network allowlist, confirmed via direct
request -> host_not_allowed). So for 2026 teams the only comparable signals
are FIFA ranking trajectory and opponent-weighted pre-tournament form
(data/external_signals.py) - the same 4 columns used for both 2018/2022
historical teams and 2026 teams, so the comparison is apples-to-apples.

Why signals are percentile-ranked within their own tournament year before
comparing: FIFA changed its ranking-points formula in Aug 2018, inflating
raw point totals from then on (2018 teams averaged ~930 points pre-tournament,
2022 teams ~1599, on the SAME underlying strength scale). Comparing raw
values made 48/48 2026 teams gravitate toward 2022 twins almost by
construction (43/48), regardless of actual resemblance - a scale artifact,
not real signal. Ranking each signal within its own year first (e.g. "this
team was in the 90th percentile of its year's ranking") removes that
artifact and lets 2018 twins compete fairly again.
"""
import pandas as pd

from data.external_signals import build_external_features
from models.similarity_engine import most_similar

PROCESSED = "data/processed"

REDUCED_COLS = [
    "fifa_points_pre",
    "fifa_points_trend_12m",
    "weighted_form_score",
    "weighted_goal_diff",
]

TEAMS_2026 = [
    "Algeria", "Argentina", "Australia", "Austria", "Belgium", "Bosnia and Herzegovina",
    "Brazil", "Canada", "Cape Verde", "Colombia", "Croatia", "Curaçao", "Czech Republic",
    "DR Congo", "Ecuador", "Egypt", "England", "France", "Germany", "Ghana", "Haiti",
    "Iran", "Iraq", "Ivory Coast", "Japan", "Jordan", "Mexico", "Morocco", "Netherlands",
    "New Zealand", "Norway", "Panama", "Paraguay", "Portugal", "Qatar", "Saudi Arabia",
    "Scotland", "Senegal", "South Africa", "South Korea", "Spain", "Sweden", "Switzerland",
    "Tunisia", "Turkey", "United States", "Uruguay", "Uzbekistan",
]

ROUND_NAMES = {0: "Grupos", 1: "Octavos", 2: "Cuartos", 3: "Semis", 4: "Final", 5: "Campeón"}


def add_year_percentile_cols(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Rank each column within its own `year` group (0-100), so 2018/2022/2026
    are compared on relative standing, not on raw point totals that live on
    different scales because of the 2018 FIFA ranking formula change."""
    out = df.copy()
    for col in cols:
        out[f"{col}_pctile"] = df.groupby("year")[col].rank(pct=True) * 100
    return out


def build_2026_features() -> pd.DataFrame:
    pairs = [(team, 2026) for team in TEAMS_2026]
    ext = build_external_features(pairs)
    missing = ext[ext["fifa_points_pre"].isna() | ext["weighted_form_score"].isna()]
    if not missing.empty:
        print(f"WARNING: missing external features for: {missing['team'].tolist()}")
    return ext


def build_comparisons(vectors: pd.DataFrame) -> pd.DataFrame:
    """For each 2026 team, find its closest historical twin (2018 or 2022)
    using only the 4 signals available for both eras, percentile-ranked
    within year. Returns a comparison table - no predicted round, no
    probabilities."""
    features_2026 = build_2026_features()
    features_2026["round_reached"] = pd.NA  # unknown; only used as source-side twin metadata
    pool_cols = ["team", "year", "round_reached"] + REDUCED_COLS
    combined = pd.concat([vectors[pool_cols], features_2026[pool_cols]], ignore_index=True)
    combined = add_year_percentile_cols(combined, REDUCED_COLS)
    pctile_cols = [f"{c}_pctile" for c in REDUCED_COLS]

    target = combined[combined["year"] == 2026].reset_index(drop=True)
    source = combined[combined["year"] != 2026].reset_index(drop=True)
    twins = most_similar(target, source, pctile_cols, top_n=1)

    merged = features_2026.merge(
        twins[["team", "twin_team", "twin_year", "similarity_pct", "twin_round_reached"]],
        on="team",
    )
    merged["twin_round_reached"] = merged["twin_round_reached"].astype(int)
    merged["twin_round_label"] = merged["twin_round_reached"].map(ROUND_NAMES)
    return merged


if __name__ == "__main__":
    vectors = pd.read_csv(f"{PROCESSED}/team_vectors.csv")
    comparisons = build_comparisons(vectors)
    comparisons = comparisons.sort_values("similarity_pct", ascending=False)
    comparisons.to_csv(f"{PROCESSED}/comparisons_2026.csv", index=False)

    cols_show = ["team", "twin_team", "twin_year", "similarity_pct",
                 "fifa_points_pre", "weighted_form_score", "twin_round_label"]
    print("\n=== Comparaciones 2026 vs. gemelo histórico más parecido (48 equipos) ===")
    print(comparisons[cols_show].to_string(index=False))

    print("\n=== Equipos solicitados ===")
    highlight = comparisons[comparisons["team"].isin(["Mexico", "Argentina", "Brazil", "Spain", "France"])]
    print(highlight[cols_show].to_string(index=False))
