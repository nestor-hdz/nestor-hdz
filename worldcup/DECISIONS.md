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

## Resultados Tarea 3 — Modelo XGBoost (round_reached)

Validación: leave-one-tournament-out, igual protocolo que el backtesting
(train 2018 → test 2022, train 2022 → test 2018), para comparar de forma
justa contra el baseline de similitud ya establecido (0.656).

Features: vector estandarizado (sin `matches_played`/`goals_for`/`goals_against`)
+ `twin_round_reached` y `similarity_pct` del gemelo histórico (calculados con
el mismo `most_similar()` del motor de similitud) + `fifa_points_pre` /
`fifa_points_trend_12m` / `weighted_form_score` / `weighted_goal_diff`.
Hiperparámetros conservadores (`max_depth=1`, `n_estimators=30`,
`reg_lambda=3.0`) por el tamaño de muestra extremadamente chico (32 filas de
entrenamiento por fold, ~17 features candidatas) — un árbol más profundo
sobreajusta de inmediato con tan pocos datos.

| Modelo | MAE (rondas) |
|---|---|
| Baseline FIFA-ranking puro | 1.172 |
| Baseline trivial ("todos a grupos") | 0.969 |
| Motor de similitud (gemelo histórico) | 0.656 |
| **XGBoost (vector completo 2018+2022)** | **0.653** |

XGBoost mejora al motor de similitud por solo 0.003 rondas — es una mejora
**marginal, no una victoria contundente**. Se reporta así explícitamente para
no repetir el error de presentar el resultado como mejor de lo que es: con 64
filas de entrenamiento el margen de error del MAE mismo es más ancho que esa
diferencia, así que XGBoost y el motor de similitud son, en la práctica,
estadísticamente equivalentes en esta muestra. La razón de mantener XGBoost
de todos modos es que combina el gemelo histórico CON el ranking FIFA y la
forma ponderada en un solo modelo, en vez de tratarlos por separado.

Modelo final entrenado en las 64 filas (2018+2022) guardado en
`models/xgb_round_model.joblib`. Reproducir con `python -m models.xgb_model`.

## Predicciones 2026 → reemplazadas por comparaciones puras (decisión explícita del usuario)

Primera versión de esta sección entrenaba un modelo XGBoost reducido para
2026 y publicaba `predicted_round`, `prob_round_0..5` y `prob_champion` en la
app. El usuario paró esto explícitamente:

> "esto no es un modelo predictorio, esto no es un bot de apuestas, aquí no
> voy a predecir quién va a ganar y quién va a perder el mundial... solo
> quiero que vayas haciendo COMPARACIONES, solo eso"

Es la misma salvaguarda #3/#4 del inicio del proyecto aplicada en la
práctica: una probabilidad de campeón, aunque esté honestamente calculada y
con su margen de error explicado, se lee como certeza — exactamente el
problema que originó este proyecto. **Se eliminó por completo** el modelo de
regresión para 2026 (`XGBRegressor`, `predicted_round`, `prob_round_*`,
`prob_champion`) y el artefacto `models/xgb_2026_reduced_model.joblib`.
`models/predict_2026.py` ahora solo hace lo que dice Tarea 2: encontrar el
gemelo histórico más parecido y mostrar el % de similitud — sin proyectar
nada hacia el futuro del equipo 2026. El modelo XGBoost completo
(`xgb_round_model.joblib`, MAE 0.653) se conserva solo como **validación de
metodología** (demuestra que el vector de features tiene señal real vs. un
baseline), no se usa para generar ningún output de cara al usuario sobre
2026.

### Transparencia de señales: qué se compara y con qué

| Comparación | Columnas usadas |
|---|---|
| 2018 ↔ 2022 (Tarea 2/3, datos completos) | `avg_possession_pct`, `avg_xg_for/against`, `avg_shots_on_target`, `avg_pass_completion_pct`, `avg_ppda`, `avg_high_pressures`, `star_goals_plus_assists`, `star_key_passes_per90`, `star_xg_plus_xa`, `star_duel_win_pct`, `fifa_points_pre`, `fifa_points_trend_12m`, `weighted_form_score`, `weighted_goal_diff` (17 columnas) |
| 2026 vs. histórico (Tarea 2 reducida) | Solo `fifa_points_pre`, `fifa_points_trend_12m`, `weighted_form_score`, `weighted_goal_diff` (4 columnas) — StatsBomb no tiene datos de evento para 2026 (torneo no jugado completo) y FBref/Transfermarkt/ESPN están bloqueados por el allowlist de red del sandbox (`host_not_allowed`, confirmado por request directo) |

Siempre excluidos de cualquier comparación (fuga de información): `matches_played`,
`goals_for`, `goals_against`, `round_reached`. Método: similitud coseno sobre
vector estandarizado (`sklearn.preprocessing.StandardScaler` + `cosine_similarity`).

### Bug encontrado y corregido: casi todos los gemelos salían de 2022

Con las 4 columnas reducidas sin normalizar, 43 de 48 equipos 2026
encontraban su gemelo en 2022 y solo 5 en 2018 — no por parecido futbolístico
real, sino porque **FIFA cambió su fórmula de puntos en agosto 2018**: los
puntos pre-torneo de 2018 promedian ~930, los de 2022 ~1599 (misma escala de
fuerza real, números completamente distintos por el cambio de fórmula). Como
2026 usa la fórmula nueva (misma escala que 2022), la comparación por valor
crudo empujaba casi todo hacia gemelos de 2022 por artefacto de escala, no
por señal.

**Fix**: cada señal se convierte a percentil dentro de su propio año
(`groupby("year").rank(pct=True)`) antes de comparar entre eras — "este
equipo estaba en el percentil 90 de su año" es comparable entre 2018, 2022 y
2026; el valor crudo no lo es. Después del fix, la distribución de gemelos
quedó 28 de 2018 / 20 de 2022 — mucho más equilibrada y consistente con que
el parecido real no debería depender del año.

Reproducir con `python -m models.predict_2026`. Output:
`data/processed/comparisons_2026.csv` (equipo, gemelo histórico + año, %
similitud, ronda que alcanzó el gemelo — sin ronda predicha ni probabilidad
para el equipo 2026).
