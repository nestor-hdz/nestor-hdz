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

## Predicciones 2026 — modelo de features reducidas (limitación de entorno)

**Restricción técnica descubierta y confirmada**: FBref, Transfermarkt y ESPN
(las fuentes planeadas para datos de partidos 2026 en curso) están bloqueadas
por el allowlist de red del sandbox (`host_not_allowed` en cada intento
directo). No hay manera de obtener para 2026 los datos a nivel de evento de
StatsBomb (xG, posesión, PPDA, stats de jugadores estrella) — ese torneo
simplemente no tiene datos StatsBomb porque no se ha jugado completo.

**Decisión**: en vez de rellenar esas columnas con valores imputados o
promedios (lo cual las dejaría dominar el modelo, o ser ignoradas, por
razones falsas — un valor inventado no es señal), se entrenó un **modelo
separado** (`models/predict_2026.py`) restringido exactamente a las columnas
que sí existen para 2026: `fifa_points_pre`, `fifa_points_trend_12m`,
`weighted_form_score`, `weighted_goal_diff` (esta última usando resultados
reales de la fase de grupos 2026 ya jugada, vía `international_results.csv`)
más el gemelo histórico calculado con ese mismo subconjunto de columnas. Así
las features de entrenamiento y de predicción coinciden exactamente.

| Modelo | MAE (rondas) |
|---|---|
| Motor de similitud (vector completo) | 0.656 |
| XGBoost (vector completo) | 0.653 |
| **XGBoost reducido (solo columnas disponibles para 2026)** | **1.002** |

La brecha (1.002 vs 0.653) **es el costo real de no tener datos de evento
2026** — no es un error de código, es la limitación honesta de la fuente de
datos. Con `max_depth=1` y solo 4 features numéricas, el modelo reducido
distingue poco entre equipos: casi todas las 48 selecciones quedaron
agrupadas en la etiqueta "Octavos" tras redondear, lo cual es consistente con
ese MAE más alto — el modelo reducido tiene mucha menos capacidad de
discriminación que el completo.

Equipos solicitados (ronda predicha, sin redondear, antes de la capa de
ajuste experto pendiente — salvaguarda #3):

| Equipo | Gemelo histórico | Similitud | Ronda predicha (cruda) |
|---|---|---|---|
| Argentina | Brasil 2018/2022 | 97.9% | 1.30 (~Octavos) |
| Brasil | Estados Unidos | 95.8% | 1.30 (~Octavos) |
| Francia | Dinamarca | 98.9% | 1.30 (~Octavos) |
| España | Brasil | 98.7% | 1.30 (~Octavos) |
| México | Irán | 95.9% | 0.68 (~Octavos) |

**Esto NO debe leerse como certeza** (salvaguarda #4, pendiente de aplicarse
formalmente en la Tarea 4/app): el modelo reducido tiene MAE ~1 ronda, así
que una diferencia de 0.6 rondas entre México y Argentina aquí está dentro
del margen de error del modelo, no es una distinción confiable. El ranking
completo de 48 equipos está en `data/processed/predictions_2026.csv`.

Modelo reducido guardado en `models/xgb_2026_reduced_model.joblib`.
Reproducir con `python -m models.predict_2026`.
