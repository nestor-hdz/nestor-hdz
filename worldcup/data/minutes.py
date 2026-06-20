"""Computes minutes played per player per match from lineup position intervals."""
import pandas as pd

PERIOD_OFFSET_MIN = {1: 0, 2: 45, 3: 90, 4: 105, 5: 120}


def _mmss_to_minutes(value: str) -> float:
    mm, ss = value.split(":")
    return int(mm) + int(ss) / 60


def get_period_end_minutes(events: pd.DataFrame) -> dict[int, float]:
    """Real (stoppage-inclusive) end minute of each period, from Half End events."""
    half_end = events[events["type"] == "Half End"]
    return half_end.groupby("period")["minute"].max().to_dict()


def player_minutes(lineup_df: pd.DataFrame, period_end_minutes: dict[int, float]) -> pd.Series:
    """Total minutes played per player_id for one team in one match."""
    records = []
    for _, row in lineup_df.iterrows():
        total = 0.0
        for pos in row["positions"]:
            period = pos["from_period"]
            period_start_offset = PERIOD_OFFSET_MIN.get(period, 0)
            start_min = period_start_offset + _mmss_to_minutes(pos["from"])
            if pos["to"] is None:
                end_period = pos["to_period"] or period
                end_min = period_end_minutes.get(
                    end_period, PERIOD_OFFSET_MIN.get(end_period, period_start_offset) + 45
                )
            else:
                end_period = pos["to_period"] or period
                end_period_offset = PERIOD_OFFSET_MIN.get(end_period, 0)
                end_min = end_period_offset + _mmss_to_minutes(pos["to"])
            total += max(0.0, end_min - start_min)
        records.append((row["player_id"], row["player_name"], total))
    out = pd.DataFrame(records, columns=["player_id", "player_name", "minutes"])
    return out.groupby(["player_id", "player_name"])["minutes"].sum()
