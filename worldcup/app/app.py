"""Task 4: Streamlit app — World Cup 2026 predictions.

Reads the already-computed predictions (models/predict_2026.py), it does not
retrain anything. Two pages: a full 48-team ranking, and a per-team detail
view with the round-probability distribution and an explicit uncertainty
warning, per the methodology safeguards in DECISIONS.md (no output here
should read as a certain forecast).
"""
import os

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

PROCESSED = os.path.join(os.path.dirname(__file__), "..", "data", "processed")

ROUND_NAMES = {0: "Grupos", 1: "Octavos", 2: "Cuartos", 3: "Semis", 4: "Final", 5: "Campeón"}
REDUCED_MODEL_MAE = 1.002  # leave-one-tournament-out MAE, see DECISIONS.md


@st.cache_data
def load_predictions() -> pd.DataFrame:
    return pd.read_csv(f"{PROCESSED}/predictions_2026.csv")


def ranking_page(preds: pd.DataFrame):
    st.header("Ranking general — 48 equipos, Mundial 2026")
    st.caption(
        "Ordenado por probabilidad de campeón. MAE del modelo: "
        f"~{REDUCED_MODEL_MAE:.1f} rondas (ver advertencia en la página de detalle)."
    )

    table = preds[[
        "team", "predicted_round_label", "twin_team", "twin_year",
        "similarity_pct", "prob_champion",
    ]].rename(columns={
        "team": "Equipo",
        "predicted_round_label": "Ronda predicha",
        "twin_team": "Gemelo histórico",
        "twin_year": "Año del gemelo",
        "similarity_pct": "% similitud",
        "prob_champion": "Prob. campeón",
    })
    table["% similitud"] = table["% similitud"].round(1)
    table["Prob. campeón"] = (table["Prob. campeón"] * 100).round(2)
    table = table.sort_values("Prob. campeón", ascending=False).reset_index(drop=True)

    st.dataframe(
        table,
        use_container_width=True,
        column_config={
            "% similitud": st.column_config.NumberColumn(format="%.1f%%"),
            "Prob. campeón": st.column_config.NumberColumn(format="%.2f%%"),
        },
        hide_index=True,
    )


def detail_page(preds: pd.DataFrame):
    st.header("Detalle por equipo")

    team = st.selectbox("Selecciona un equipo", sorted(preds["team"].unique()))
    row = preds[preds["team"] == team].iloc[0]

    st.markdown(
        f"### {team} 2026 se parece en **{row['similarity_pct']:.1f}%** "
        f"a **{row['twin_team']} {int(row['twin_year'])}**"
    )
    st.caption(
        f"Ese gemelo histórico llegó a: {ROUND_NAMES[int(row['twin_round_reached'])]} "
        f"({int(row['twin_round_reached'])})."
    )

    round_labels = [ROUND_NAMES[r] for r in range(6)]
    probs = [row[f"prob_round_{r}"] * 100 for r in range(6)]

    fig = go.Figure(go.Bar(x=round_labels, y=probs, marker_color="#1f77b4"))
    fig.update_layout(
        title=f"Probabilidad de llegar a cada ronda — {team}",
        xaxis_title="Ronda",
        yaxis_title="Probabilidad (%)",
        yaxis_range=[0, 100],
    )
    st.plotly_chart(fig, use_container_width=True)

    st.warning(
        "**Esto no es una certeza.** El modelo reducido usado para 2026 "
        f"(ranking FIFA + forma pre-torneo ponderada) tiene un error histórico "
        f"de ~{REDUCED_MODEL_MAE:.1f} ronda en validación leave-one-tournament-out "
        "(2018↔2022) — más alto que el modelo completo (0.65), porque a 2026 le "
        "faltan los datos de partido en curso (xG, posesión, etc. — StatsBomb no "
        "los tiene todavía y FBref/Transfermarkt/ESPN no son accesibles desde "
        "este entorno). Las probabilidades de arriba son una distribución "
        "estimada centrada en la predicción del modelo, con un ancho calibrado "
        "por ese mismo error histórico — no un pronóstico exacto. Diferencias "
        "chicas entre equipos (menos de ~1 ronda) no son distinciones confiables. "
        "Ver DECISIONS.md para la metodología completa y sus límites."
    )


def main():
    st.set_page_config(page_title="Mundial 2026 — Predicciones", layout="wide")
    st.title("Mundial 2026: equipos vs. gemelos históricos")

    preds = load_predictions()

    page = st.sidebar.radio("Página", ["Ranking general", "Detalle por equipo"])
    if page == "Ranking general":
        ranking_page(preds)
    else:
        detail_page(preds)


if __name__ == "__main__":
    main()
