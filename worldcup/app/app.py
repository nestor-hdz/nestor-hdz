"""Task 4: Streamlit app — World Cup 2026 comparisons.

This is a COMPARISON tool, not a predictor. It reads the output of
models/predict_2026.py (comparisons_2026.csv): for each 2026 team, its
closest historical twin (2018 or 2022) by FIFA ranking + pre-tournament
form, and what that twin went on to do. There is no predicted round, no
round-by-round probability, and no championship probability anywhere in
this app - an earlier version of this project did present a similarity
score as a confident forecast and that was the wrong call (see
DECISIONS.md). Two pages: a 48-team comparison table, and a per-team detail
view with a signal-by-signal comparison chart against the twin.
"""
import os

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

PROCESSED = os.path.join(os.path.dirname(__file__), "..", "data", "processed")

ROUND_NAMES = {0: "Grupos", 1: "Octavos", 2: "Cuartos", 3: "Semis", 4: "Final", 5: "Campeón"}
SIGNAL_LABELS = {
    "fifa_points_pre": "Puntos FIFA pre-torneo",
    "fifa_points_trend_12m": "Tendencia FIFA (12 meses)",
    "weighted_form_score": "Forma ponderada por rival",
    "weighted_goal_diff": "Diferencia de gol ponderada",
}


@st.cache_data
def load_comparisons() -> pd.DataFrame:
    return pd.read_csv(f"{PROCESSED}/comparisons_2026.csv")


def ranking_page(comps: pd.DataFrame):
    st.header("Comparación general — 48 equipos, Mundial 2026")
    st.caption(
        "Cada equipo 2026 comparado con su gemelo histórico (2018 o 2022) más "
        "parecido, por ranking FIFA y forma pre-torneo. Esto NO es una "
        "predicción de qué tan lejos llegará cada equipo — es una analogía "
        "estadística. Ordenado por % de similitud descendente."
    )

    table = comps[[
        "team", "twin_team", "twin_year", "similarity_pct", "twin_round_label",
    ]].rename(columns={
        "team": "Equipo 2026",
        "twin_team": "Gemelo histórico",
        "twin_year": "Año del gemelo",
        "similarity_pct": "% similitud",
        "twin_round_label": "Ronda a la que llegó el gemelo",
    })
    table["% similitud"] = table["% similitud"].round(1)
    table = table.sort_values("% similitud", ascending=False).reset_index(drop=True)

    st.dataframe(
        table,
        use_container_width=True,
        column_config={"% similitud": st.column_config.NumberColumn(format="%.1f%%")},
        hide_index=True,
    )


def detail_page(comps: pd.DataFrame):
    st.header("Detalle por equipo")

    team = st.selectbox("Selecciona un equipo", sorted(comps["team"].unique()))
    row = comps[comps["team"] == team].iloc[0]

    st.markdown(
        f"### {team} 2026 se parece en **{row['similarity_pct']:.1f}%** "
        f"a **{row['twin_team']} {int(row['twin_year'])}**"
    )
    st.caption(
        f"Ese gemelo histórico llegó a: {row['twin_round_label']}. "
        "Esto es contexto histórico, no una proyección de que el equipo 2026 "
        "vaya a llegar a la misma ronda."
    )

    signal_cols = list(SIGNAL_LABELS.keys())
    fig = go.Figure()
    fig.add_bar(name=f"{team} 2026", x=[SIGNAL_LABELS[c] for c in signal_cols],
                y=[row[c] for c in signal_cols], marker_color="#1f77b4")
    fig.update_layout(
        title=f"Señales comparadas: {team} 2026 vs. {row['twin_team']} {int(row['twin_year'])}",
        yaxis_title="Valor de la señal",
        barmode="group",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.warning(
        "**Esto es una comparación, no una predicción.** Solo se usan 4 "
        "señales para 2026 (ranking FIFA y forma pre-torneo ponderada por "
        "rival) porque StatsBomb no tiene datos de partido para este torneo "
        "todavía y FBref/Transfermarkt/ESPN no son accesibles desde este "
        "entorno — es una base mucho más angosta que la usada para comparar "
        "2018 contra 2022 (que incluye posesión, xG, presión y estadísticas "
        "de jugadores clave). El % de similitud describe qué tan parecido es "
        "el equipo 2026 a su gemelo histórico en esas señales — no dice nada "
        "sobre qué ronda alcanzará el equipo 2026. Ver DECISIONS.md para la "
        "metodología completa y sus límites."
    )


def main():
    st.set_page_config(page_title="Mundial 2026 — Comparaciones", layout="wide")
    st.title("Mundial 2026: equipos vs. gemelos históricos")

    comps = load_comparisons()

    page = st.sidebar.radio("Página", ["Comparación general", "Detalle por equipo"])
    if page == "Comparación general":
        ranking_page(comps)
    else:
        detail_page(comps)


if __name__ == "__main__":
    main()
