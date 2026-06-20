"""2026 predictions using a REDUCED feature model.

Why reduced: StatsBomb has no event-level data for World Cup 2026 (it
hasn't been played in full, and the FBref scraper from Task 1.5 can't run in
this sandbox - fbref.com/transfermarkt.com/espn.com are all blocked by the
network allowlist, confirmed via direct request, host_not_allowed). So for
2026 teams we only have:
  - FIFA ranking points, but the dataset's last snapshot is Sept 2024 (~21
    months stale by the 2026 tournament) - NOT a live ranking.
  - Real 2026 group-stage results played so far (international_results.csv),
    used for opponent-weighted pre-tournament form.

Feeding StatsBomb-only columns (xG, possession, PPDA, star-player stats) as
imputed constants into the full model trained in xgb_model.py would let
those columns silently dominate or be ignored for the wrong reasons. Instead
this trains a SEPARATE model on 2018+2022 restricted to the same reduced
feature space available for 2026, so train-time and predict-time features
match exactly. Its own leave-one-tournament-out MAE is reported alongside
the full model's, and it is visibly worse - that gap IS the cost of missing
in-tournament data for 2026, not a hidden one.
"""
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor

from data.external_signals import build_external_features, RANKING_NAME_MAP
from models.similarity_engine import most_similar

PROCESSED = "data/processed"
MODELS_DIR = "models"

REDUCED_COLS = [
    "fifa_points_pre",
    "fifa_points_trend_12m",
    "weighted_form_score",
    "weighted_goal_diff",
]
REDUCED_PARAMS = dict(n_estimators=20, max_depth=1, learning_rate=0.15, reg_lambda=3.0, random_state=42)

TEAMS_2026 = [
    "Algeria", "Argentina", "Australia", "Austria", "Belgium", "Bosnia and Herzegovina",
    "Brazil", "Canada", "Cape Verde", "Colombia", "Croatia", "Curaçao", "Czech Republic",
    "DR Congo", "Ecuador", "Egypt", "England", "France", "Germany", "Ghana", "Haiti",
    "Iran", "Iraq", "Ivory Coast", "Japan", "Jordan", "Mexico", "Morocco", "Netherlands",
    "New Zealand", "Norway", "Panama", "Paraguay", "Portugal", "Qatar", "Saudi Arabia",
    "Scotland", "Senegal", "South Africa", "South Korea", "Spain", "Sweden", "Switzerland",
    "Tunisia", "Turkey", "United States", "Uruguay", "Uzbekistan",
]


def reduced_training_table(vectors: pd.DataFrame) -> pd.DataFrame:
    """2018+2022 rows with twin features computed using ONLY the reduced
    columns, so the twin-finding logic matches what's actually possible for
    2026 teams (no advanced in-tournament stats)."""
    rows = []
    for test_year in (2018, 2022):
        target = vectors[vectors["year"] == test_year].reset_index(drop=True)
        source = vectors[vectors["year"] != test_year].reset_index(drop=True)
        twins = most_similar(target, source, REDUCED_COLS, top_n=1)
        merged = target.merge(
            twins[["team", "similarity_pct", "twin_round_reached"]], on="team"
        )
        rows.append(merged)
    return pd.concat(rows, ignore_index=True)


def reduced_loto_eval(table: pd.DataFrame) -> float:
    cols = REDUCED_COLS + ["twin_round_reached", "similarity_pct"]
    maes = []
    for test_year in (2018, 2022):
        train = table[table["year"] != test_year]
        test = table[table["year"] == test_year]
        model = XGBRegressor(**REDUCED_PARAMS)
        model.fit(train[cols], train["round_reached"])
        pred = model.predict(test[cols])
        maes.append(mean_absolute_error(test["round_reached"], pred))
    return float(np.mean(maes))


def build_2026_features() -> pd.DataFrame:
    pairs = [(team, 2026) for team in TEAMS_2026]
    ext = build_external_features(pairs)
    missing = ext[ext["fifa_points_pre"].isna() | ext["weighted_form_score"].isna()]
    if not missing.empty:
        print(f"WARNING: missing external features for: {missing['team'].tolist()}")
    return ext


def predict_2026(vectors: pd.DataFrame, reduced_table: pd.DataFrame) -> pd.DataFrame:
    cols = REDUCED_COLS + ["twin_round_reached", "similarity_pct"]
    model = XGBRegressor(**REDUCED_PARAMS)
    model.fit(reduced_table[cols], reduced_table["round_reached"])

    features_2026 = build_2026_features()
    source = vectors  # all 64 historical teams as the twin-search pool
    twins = most_similar(features_2026, source, REDUCED_COLS, top_n=1)
    merged = features_2026.merge(
        twins[["team", "twin_team", "similarity_pct", "twin_round_reached"]], on="team"
    )

    merged["predicted_round"] = model.predict(merged[cols])
    merged["predicted_round_rounded"] = np.clip(np.round(merged["predicted_round"]), 0, 5).astype(int)
    return merged


ROUND_NAMES = {0: "Grupos", 1: "Octavos", 2: "Cuartos", 3: "Semis", 4: "Final", 5: "Campeón"}


if __name__ == "__main__":
    vectors = pd.read_csv(f"{PROCESSED}/team_vectors.csv")
    reduced_table = reduced_training_table(vectors)
    reduced_mae = reduced_loto_eval(reduced_table)
    print(f"Reduced-feature model LOTO MAE (2018<->2022): {reduced_mae:.3f} rounds")
    print("(Full model MAE: 0.653 | Similarity-engine baseline: 0.656 -- "
          "this gap is the cost of not having 2026 in-tournament stats)")

    preds = predict_2026(vectors, reduced_table)
    preds["predicted_round_label"] = preds["predicted_round_rounded"].map(ROUND_NAMES)
    preds = preds.sort_values("predicted_round", ascending=False)
    preds.to_csv(f"{PROCESSED}/predictions_2026.csv", index=False)

    joblib.dump(
        {"model": XGBRegressor(**REDUCED_PARAMS).fit(reduced_table[REDUCED_COLS + ["twin_round_reached", "similarity_pct"]], reduced_table["round_reached"]), "feature_cols": REDUCED_COLS + ["twin_round_reached", "similarity_pct"]},
        f"{MODELS_DIR}/xgb_2026_reduced_model.joblib",
    )

    cols_show = ["team", "twin_team", "similarity_pct", "fifa_points_pre", "weighted_form_score",
                 "predicted_round", "predicted_round_label"]
    print("\n=== Full 48-team ranking ===")
    print(preds[cols_show].to_string(index=False))

    print("\n=== Equipos solicitados ===")
    highlight = preds[preds["team"].isin(["Mexico", "Argentina", "Brazil", "Spain", "France"])]
    print(highlight[cols_show].to_string(index=False))
