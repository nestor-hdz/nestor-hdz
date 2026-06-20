# Decisiones de metodología — World Cup 2026 Analysis

Registro de lo acordado en discusión, para que no se pierda entre sesiones.

## Problema detectado (sesión anterior, proyecto similar)

Un análisis previo basado solo en datos históricos predijo un campeón poco
creíble y se presentó como certeza, no como hipótesis. Causas identificadas:

1. **Sin ponderación por fuerza de rival** — un gol contra un equipo débil
   contaba igual que uno contra un equipo fuerte.
2. **Sin contexto cualitativo** — lesiones, cambios de táctica, dificultad
   del grupo/bracket no entraban al análisis.
3. **Output comunicado como certeza** — un % de similitud o probabilidad se
   leyó como predicción firme, sin dejar claro el margen de error ni el
   tamaño de muestra (64 equipos-torneo en total para 2018+2022).

## Salvaguardas acordadas

| # | Salvaguarda | Estado |
|---|---|---|
| 1 | **Ponderación por fuerza de rival** en señales de forma pre-torneo (`weighted_form_score`, `weighted_goal_diff` en `data/external_signals.py`): cada resultado se pesa por los puntos FIFA del rival al momento del partido. | ✅ Implementado |
| 2 | **Backtesting obligatorio** antes de confiar en el modelo para 2026: entrenar con 2018 y validar contra 2022 (y viceversa) antes de usarlo en datos reales. | ✅ Implementado — ver resultados abajo |
| 3 | **Capa de ajuste experto/cualitativo**: campos editables por un humano (lesiones, continuidad técnica, forma reciente, dificultad de grupo) que modulan el output estadístico. | ⏳ Pendiente — explícitamente para después, se usará como feedback humano sobre el modelo ya construido |
| 4 | **Comunicación de incertidumbre**: ningún output final dice "X será campeón"; siempre "el modelo da Y%, basado en N precedentes, con margen de error amplio". | ⏳ Pendiente — se aplica al construir la Tarea 4 (app) |

## Señales nuevas incorporadas (Task 1.5)

Fuentes públicas, sin scraping ni credenciales de pago:

- **`ranking_fifa_historical.csv`** (github.com/Dato-Futbol/fifa-ranking,
  1992–2024): puntos FIFA de cada equipo en el tiempo. Se usa para:
  - `fifa_points_pre`: puntos FIFA justo antes de que arrancara el torneo.
  - `fifa_points_trend_12m`: tendencia (¿subía o bajaba el equipo en el año
    previo?) — captura trayectoria, no solo una foto fija.
- **`international_results.csv`** (github.com/martj42/international_results,
  1872–2026, incluye resultados reales de la fase de grupos 2026 ya jugada):
  se usa para `weighted_form_score` / `weighted_goal_diff` — forma en los
  últimos 15 partidos antes del torneo, ponderada por la fuerza FIFA del
  rival enfrentado.

Mapeo de nombres necesario entre StatsBomb y estas fuentes:
`Iran→IR Iran`, `South Korea→Korea Republic`, `United States→USA`
(ver `data/external_signals.py:NAME_MAP`).

## Señales identificadas pero no implementadas (backlog)

- Valor de mercado del plantel (Transfermarkt) — requiere scraping.
- Continuidad/cambios de cuerpo técnico.
- Lesiones de jugadores clave durante el torneo.
- Impacto de sustituciones, eficiencia en balón parado, varianza de
  rendimiento partido a partido — señales derivadas propias, pendientes de
  diseño y validación de que aportan algo (no añadir por añadir).

## Resultados del backtesting (2018↔2022)

Motor: cosine similarity sobre vector estandarizado (excluyendo
`matches_played`/`goals_for`/`goals_against` por fuga de información — están
mecánicamente ligados a cuántas rondas jugó el equipo). Predicción = ronda
del "gemelo" histórico más similar. Comparado contra dos baselines: (a)
vecino más cercano por ranking FIFA puro, (b) baseline trivial "todos quedan
en grupos" (justo porque 16/32 equipos siempre quedan ahí, un baseline ingenuo
puede parecer bueno sin aportar nada).

| Dirección | MAE modelo | MAE baseline FIFA | MAE baseline trivial |
|---|---|---|---|
| 2022 ← gemelos 2018 | 0.625 | 1.344 | 0.969 |
| 2018 ← gemelos 2022 | 0.688 | 1.000 | 0.969 |
| **Combinado** | **0.656** | **1.172** | **0.969** |

**El modelo supera a ambos baselines** (margen ~0.3 rondas vs el mejor
baseline) → hay señal real en el vector de similitud, no es solo ruido.

**Pero no es perfecto, y eso es justo el punto**: Suiza (2022) salió
predicha como campeona (gemelo: Francia 2018, 95%+ similar en xG/posesión)
y en la realidad quedó eliminada en octavos — error de 4 rondas, el peor
caso individual. Argentina (campeón real 2022) salió predicha en semis
(gemelo: Croacia 2018) — error de 1 ronda, subestimada. Esto confirma lo
discutido: la similitud estadística pura puede acertar en promedio pero
fallar feo en casos individuales — exactamente donde debe entrar la capa de
ajuste experto (salvaguarda #3) antes de presentar una predicción de 2026
como definitiva.

Tablas completas: `data/processed/backtest_2022_from_2018.csv` y
`data/processed/backtest_2018_from_2022.csv`. Reproducir con
`python -m models.backtest`.

## Próximo paso acordado

Backtesting confirma señal real pero con outliers grandes (Suiza, Argentina).
Antes de construir el modelo final de Tarea 3 para 2026, evaluar si conviene
ya incorporar la capa de ajuste experto (salvaguarda #3) en paralelo, dado
que el backtest mostró casos concretos donde el puro dato se equivoca feo.
