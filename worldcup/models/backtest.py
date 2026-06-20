"""Backtest: does the cosine-similarity engine actually predict round_reached
better than a naive baseline, on a tournament we already know the answer to?

For every 2022 team: find its most similar 2018 "twin" (full standardized
feature vector) and use the twin's round_reached as the prediction. Compare
against a naive baseline that does the same nearest-neighbor lookup using
only `fifa_points_pre` (i.e. "just rank them by FIFA ranking"). If the full
vector doesn't beat the ranking-only baseline, the extra features aren't
adding predictive signal and the vector needs rework before touching 2026.
"""
import numpy as np
import pandas as pd

from models.similarity_engine import feature_columns, most_similar

PROCESSED = "data/processed"


def naive_ranking_match(target: pd.DataFrame, source: pd.DataFrame) -> pd.DataFrame:
    """Baseline: nearest 2018 team by absolute FIFA-points difference.

    (Not cosine similarity — cosine on a single scalar feature only captures
    sign agreement, not distance, so it's degenerate for a 1-D comparison.)
    """
    rows = []
    for _, t in target.iterrows():
        diffs = (source["fifa_points_pre"] - t["fifa_points_pre"]).abs()
        j = diffs.idxmin()
        rows.append(
            {
                "team": t["team"],
                "twin_team": source.loc[j, "team"],
                "twin_round_reached": source.loc[j, "round_reached"],
            }
        )
    return pd.DataFrame(rows)


def run_backtest(target_year: int, source_year: int) -> pd.DataFrame:
    vectors = pd.read_csv(f"{PROCESSED}/team_vectors.csv")
    target = vectors[vectors["year"] == target_year].reset_index(drop=True)
    source = vectors[vectors["year"] == source_year].reset_index(drop=True)

    full_cols = feature_columns(vectors)
    full_match = most_similar(target, source, full_cols, top_n=1)

    baseline_match = naive_ranking_match(target, source)

    out = target[["team", "round_reached"]].rename(columns={"round_reached": "actual_round"})
    out = out.merge(
        full_match[["team", "twin_team", "similarity_pct", "twin_round_reached"]].rename(
            columns={
                "twin_team": "twin_2018",
                "similarity_pct": "similarity_pct",
                "twin_round_reached": "predicted_round_model",
            }
        ),
        on="team",
    )
    out = out.merge(
        baseline_match[["team", "twin_team", "twin_round_reached"]].rename(
            columns={"twin_team": "twin_2018_baseline", "twin_round_reached": "predicted_round_baseline"}
        ),
        on="team",
    )
    majority_class = source["round_reached"].mode().iloc[0]
    out["predicted_round_trivial"] = majority_class
    out["error_model"] = (out["actual_round"] - out["predicted_round_model"]).abs()
    out["error_baseline"] = (out["actual_round"] - out["predicted_round_baseline"]).abs()
    out["error_trivial"] = (out["actual_round"] - out["predicted_round_trivial"]).abs()
    return out


def summarize(out: pd.DataFrame, target_year: int, source_year: int):
    mae_model = out["error_model"].mean()
    mae_baseline = out["error_baseline"].mean()
    mae_trivial = out["error_trivial"].mean()
    print(f"\n=== Backtest: predict {target_year} round_reached using {source_year} twins ===")
    cols = [
        "team",
        "twin_2018",
        "similarity_pct",
        "actual_round",
        "predicted_round_model",
        "error_model",
        "predicted_round_baseline",
        "error_baseline",
    ]
    print(out[cols].sort_values("error_model", ascending=False).to_string(index=False))
    print(f"\nMAE — similarity model         : {mae_model:.3f} rounds")
    print(f"MAE — naive FIFA-rank baseline  : {mae_baseline:.3f} rounds")
    print(f"MAE — trivial 'predict group stage for everyone' baseline: {mae_trivial:.3f} rounds")
    weakest = max(mae_baseline, mae_trivial)
    if mae_model < weakest:
        print(f"Model beats both baselines (margin {weakest - mae_model:.3f} rounds) -> real signal.")
    else:
        print(
            "Model does NOT clearly beat the stronger baseline -> vector needs rework "
            "before trusting it on 2026."
        )
    return mae_model, mae_baseline, mae_trivial


if __name__ == "__main__":
    out_22 = run_backtest(target_year=2022, source_year=2018)
    out_22.to_csv(f"{PROCESSED}/backtest_2022_from_2018.csv", index=False)
    mae_m1, mae_b1, mae_t1 = summarize(out_22, 2022, 2018)

    out_18 = run_backtest(target_year=2018, source_year=2022)
    out_18.to_csv(f"{PROCESSED}/backtest_2018_from_2022.csv", index=False)
    mae_m2, mae_b2, mae_t2 = summarize(out_18, 2018, 2022)

    print("\n=== Combined (both directions) ===")
    print(f"MAE — similarity model         : {np.mean([mae_m1, mae_m2]):.3f} rounds")
    print(f"MAE — naive FIFA-rank baseline  : {np.mean([mae_b1, mae_b2]):.3f} rounds")
    print(f"MAE — trivial baseline          : {np.mean([mae_t1, mae_t2]):.3f} rounds")
