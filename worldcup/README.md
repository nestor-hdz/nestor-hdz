# World Cup 2026 Analysis — MVP

Data pipeline + similarity engine + predictive model + Streamlit app comparing
World Cup 2026 teams/players to historical (2018, 2022) profiles.

## Status

- [x] **Task 1 — Data pipeline**: StatsBomb open-data (2018, 2022) team +
      star-player features, vectorized per team.
- [x] **Task 1.5 — External signals**: FIFA ranking trajectory + opponent-
      strength-weighted pre-tournament form (see DECISIONS.md for why).
- [x] **Task 2 (partial) — Similarity engine**: cosine similarity over a
      standardized vector, backtested 2018↔2022. Beats both a FIFA-ranking
      baseline and a trivial "everyone exits in groups" baseline (combined
      MAE 0.66 vs 1.17 / 0.97 rounds) — see DECISIONS.md for the full
      results, including the cases where it fails badly (Switzerland 2022).
- [x] **Task 3 — Predictive model**: XGBoost on the full 2018+2022 vector
      (leakage cols excluded) + historical-twin features, leave-one-
      tournament-out MAE 0.653 (marginal improvement over the 0.656
      similarity baseline — reported as essentially tied, not a win). A
      second, reduced-feature model (`models/predict_2026.py`) predicts the
      48 World Cup 2026 teams using only FIFA ranking + opponent-weighted
      form (StatsBomb event data doesn't exist yet for 2026, and FBref/
      Transfermarkt/ESPN are blocked by this environment's network
      allowlist) — its LOTO MAE (1.002) is honestly worse, quantifying the
      cost of missing in-tournament data. See DECISIONS.md for full results
      and the 2026 predictions.
- [ ] Task 4 — Streamlit app + deploy

See `DECISIONS.md` for the methodology safeguards agreed on (opponent-
strength weighting, mandatory backtesting, expert-adjustment layer,
uncertainty framing) after a prior pure-stats approach overstated confidence.

## Data confirmed available (StatsBomb open-data, no credentials needed)

- `competitions/matches`: World Cup 2018 (season_id=3) and 2022 (season_id=106),
  competition_id=43. Match-level `competition_stage` (Group Stage → Final) used
  to derive each team's deepest round reached (penalty shootouts resolved via
  period-5 shot outcomes, since StatsBomb doesn't expose a penalty-score field).
- `events`: `shot_statsbomb_xg`, `pass_outcome`, `possession`/`possession_team`,
  `Pressure`/`Duel`/`Interception` event types, `pass_assisted_shot_id` /
  `pass_shot_assist` (used to derive assists and xA).
- `lineups`: per-player position intervals (`from`/`to` + period) used to
  compute minutes played.
- **Not available** without paid credentials: `team_season_stats` /
  `player_season_stats` (aggregated stats endpoints) — everything here is
  computed from raw events instead.

## Layout

```
worldcup/
  data/
    statsbomb_loader.py   # cached StatsBomb open-data loader
    minutes.py            # minutes played from lineup position intervals
    team_features.py      # per-team per-tournament features + round_reached target
    player_features.py    # top-5-by-minutes player features per team
    external_signals.py    # FIFA ranking + opponent-weighted pre-tournament form
    build_dataset.py       # orchestrates -> data/processed/team_vectors.csv
    processed/             # generated CSVs (committed, small)
    raw/                   # StatsBomb cache (gitignored, ~66MB, regenerate via pipeline)
    external/              # FIFA ranking / international results (gitignored, auto-downloaded)
  models/
    similarity_engine.py  # cosine similarity, standardized vector, leakage cols excluded
    backtest.py            # 2018<->2022 validation vs FIFA-rank and trivial baselines
  app/      # Task 4
  scraping/ # FBref scraper for World Cup 2026 in-progress data (still pending)
```

## Team feature vector

Per team per tournament: `goals_for/against`, `avg_possession_pct`,
`avg_xg_for/against`, `avg_shots_on_target`, `avg_pass_completion_pct`,
`avg_ppda` (pressing intensity proxy — see below), `avg_high_pressures`,
plus the mean of the top-5-by-minutes players' `goals_plus_assists`,
`key_passes_per90`, `xg_plus_xa`, `duel_win_pct` (prefixed `star_*`).

### PPDA proxy

StatsBomb doesn't fix attacking direction in pitch coordinates, so direction
is inferred per team/period from the mean x-location of that team's shots,
then locations are flipped so every team always "attacks" toward x=120.
PPDA = opponent's completed passes in their own defensive 60% of the pitch
divided by the team's defensive actions (`Pressure`, `Duel`, `Interception`,
`Foul Committed`) in that same zone. Lower = more aggressive high press.

## External signals (Task 1.5)

Two public, no-auth datasets, auto-downloaded on first run into
`data/external/` (gitignored):

- **FIFA ranking history** (Dato-Futbol/fifa-ranking, 1992–2024) →
  `fifa_points_pre` (points right before the tournament started) and
  `fifa_points_trend_12m` (rising or declining over the prior year).
- **International results** (martj42/international_results, 1872–2026,
  includes real WC 2026 group-stage scores already played) →
  `weighted_form_score` / `weighted_goal_diff`: form over the last 15
  matches before the tournament, each result weighted by the opponent's
  FIFA points at match time — beating a top side counts more than beating
  a minnow.

## Reproduce

```
pip install -r requirements.txt
cd worldcup
python -m data.build_dataset
```

Sanity check: round_reached distribution is 32 teams at Group Stage (0),
16 at Round of 16 (1), 8 at Quarters (2), 4 at Semis (3), 2 at Final (4),
1 champion (5) — matches each tournament's bracket exactly. 2022 champion
correctly resolved as Argentina (won final on penalties); 2018 as France.
