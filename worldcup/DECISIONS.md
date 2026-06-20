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
| 2 | **Backtesting obligatorio** antes de confiar en el modelo para 2026: entrenar con 2018 y validar contra 2022 (y viceversa) antes de usarlo en datos reales. | ⏳ Pendiente — se hace cuando haya "algo considerable" construido (decisión del usuario) |
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

## Próximo paso acordado

Con la capa de señales externas ya construida, lo que sigue es backtesting
(2018↔2022) antes de construir el modelo final para 2026 — para confirmar
que el enfoque predice razonablemente un torneo conocido antes de confiar en
él para uno desconocido.
