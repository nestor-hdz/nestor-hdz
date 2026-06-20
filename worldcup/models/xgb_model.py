"""Task 3: XGBoost model for predicted round_reached.

Training table = team_vectors features (excluding leakage cols) + the
cross-tournament twin features (twin_round_reached, similarity_pct) computed
the same way as the backtest, on the combined 2018+2022 set (64 rows).

Validation: leave-one-tournament-out (train 2018 -> test 2022, train 2022 ->
test 2018), the same protocol used for the similarity-engine backtest, so the
MAE is directly comparable to the 0.656-round baseline already established.
"""
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor

from models.backtest import run_backtest
from models.similarity_engine import feature_columns

PROCESSED = "data/processed"
MODELS_DIR = os.path.dirname(__file__)

XGB_PARAMS = dict(
    n_estimators=30,
    max_depth=1,  # depth-2+ trees overfit with only 32 training rows per fold
    learning_rate=0.15,
    reg_lambda=3.0,
    random_state=42,
)


def build_training_table() -> pd.DataFrame:
    """64 rows (2018+2022), each with its own vector features plus its
    cross-tournament twin's round_reached and similarity_pct."""
    vectors = pd.read_csv(f"{PROCESSED}/team_vectors.csv")

    twins_22 = run_backtest(target_year=2022, source_year=2018)[
        ["team", "similarity_pct", "predicted_round_model"]
    ].rename(columns={"predicted_round_model": "twin_round_reached"})
    twins_22["year"] = 2022

    twins_18 = run_backtest(target_year=2018, source_year=2022)[
        ["team", "similarity_pct", "predicted_round_model"]
    ].rename(columns={"predicted_round_model": "twin_round_reached"})
    twins_18["year"] = 2018

    twins = pd.concat([twins_22, twins_18], ignore_index=True)
    table = vectors.merge(twins, on=["team", "year"], how="left")
    return table


def model_feature_columns(table: pd.DataFrame) -> list[str]:
    vectors = pd.read_csv(f"{PROCESSED}/team_vectors.csv")
    cols = feature_columns(vectors)  # vector features only, computed before twin cols were merged
    return cols + ["twin_round_reached", "similarity_pct"]


def leave_one_tournament_out_eval(table: pd.DataFrame) -> pd.DataFrame:
    cols = model_feature_columns(table)
    results = []
    for test_year in (2018, 2022):
        train = table[table["year"] != test_year]
        test = table[table["year"] == test_year]

        model = XGBRegressor(**XGB_PARAMS)
        model.fit(train[cols], train["round_reached"])
        pred = model.predict(test[cols])

        out = test[["team", "year", "round_reached"]].copy()
        out["predicted_round"] = pred
        out["predicted_round_rounded"] = np.clip(np.round(pred), 0, 5).astype(int)
        out["error"] = (out["round_reached"] - out["predicted_round"]).abs()
        results.append(out)
    return pd.concat(results, ignore_index=True)


def train_final_model(table: pd.DataFrame) -> tuple[XGBRegressor, list[str]]:
    cols = model_feature_columns(table)
    model = XGBRegressor(**XGB_PARAMS)
    model.fit(table[cols], table["round_reached"])
    return model, cols


if __name__ == "__main__":
    table = build_training_table()
    table.to_csv(f"{PROCESSED}/training_table.csv", index=False)

    eval_out = leave_one_tournament_out_eval(table)
    eval_out.to_csv(f"{PROCESSED}/xgb_loto_eval.csv", index=False)
    mae = mean_absolute_error(eval_out["round_reached"], eval_out["predicted_round"])
    print(eval_out.sort_values("error", ascending=False).to_string(index=False))
    print(f"\nXGBoost leave-one-tournament-out MAE: {mae:.3f} rounds")
    print("Similarity-engine baseline MAE (established earlier): 0.656 rounds")
    if mae < 0.656:
        print(f"XGBoost beats the similarity baseline by {0.656 - mae:.3f} rounds.")
    else:
        print(f"XGBoost does NOT beat the similarity baseline (worse by {mae - 0.656:.3f} rounds).")

    model, cols = train_final_model(table)
    joblib.dump({"model": model, "feature_cols": cols}, f"{MODELS_DIR}/xgb_round_model.joblib")
    print(f"\nFinal model trained on all 64 teams, saved to models/xgb_round_model.joblib")
