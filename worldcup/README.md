# World Cup 2026 Analysis — MVP

Data pipeline + similarity engine + predictive model + Streamlit app comparing
World Cup 2026 teams/players to historical (2018, 2022) profiles.

## Status

- [x] **Task 1 — Data pipeline**: StatsBomb open-data (2018, 2022) team +
      star-player features, vectorized per team.
- [ ] Task 2 — Similarity engine (cosine similarity, historical "twin" teams)
- [ ] Task 3 — Predictive model (XGBoost, round-reached probabilities)
- [ ] Task 4 — Streamlit app + deploy

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
    build_dataset.py       # orchestrates -> data/processed/team_vectors.csv
    processed/             # generated CSVs (committed, small)
    raw/                   # StatsBomb cache (gitignored, ~66MB, regenerate via pipeline)
  models/   # Task 3
  app/      # Task 4
  scraping/ # Task 1.5 — FBref scraper for World Cup 2026 in-progress data
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
