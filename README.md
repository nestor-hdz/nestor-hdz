# Néstor Hernández Vázquez

**Physics & Mathematics · ESFM-IPN · Quantitative Finance · AI**

I build at the intersection of rigorous mathematics and financial systems. Physics student by training, quant by direction — applying stochastic processes, numerical methods, and ML to real market problems.

Targeting top MFE programs (Princeton · CMU · Berkeley · Baruch) for 2027.

---

## Projects

### KaxaNuk Research Challenge 2026 — Quantitative Trading Strategies

> **Team leader · June 2026**

Three systematic momentum strategies developed for the KaxaNuk Quantitative Finance Research Challenge, built on the `kaxanuk-backtest-engine` framework. Each strategy targets a distinct asset class — US equities, ETFs, and crypto — and is evaluated on risk-adjusted outperformance relative to a passive benchmark.

#### Strategy Overview

**1 · US Equities Momentum — Golden Cross**

A trend-following strategy on the S&P 500 universe using a 50/200-day Simple Moving Average crossover. Enters long when SMA-50 crosses above SMA-200 (Golden Cross), exits on the inverse (Death Cross). Equally weighted across qualifying positions, rebalanced daily.

| Metric | Value |
|--------|-------|
| Total Return | +208% |
| CAGR | 19.96% |
| Alpha vs. S&P 500 | 4.57% |
| Signal | SMA 50 / 200 crossover |

---

**2 · ETF Momentum Rotation**

A cross-sectional momentum strategy that ranks a diversified ETF universe by 12-1 momentum score (12-month return minus the most recent month, to avoid short-term reversal). Allocates capital to the top-ranked ETFs each month, rotating out of laggards. Benchmarked against a buy-and-hold ETF basket.

| Metric | Value |
|--------|-------|
| Total Return | +175% |
| CAGR | 21.71% |
| Alpha vs. Benchmark | 6.92% |
| Signal | 12-1 momentum score (monthly rotation) |

---

**3 · Crypto Momentum — EMA Crossover with Inverse-ATR Sizing**

A momentum strategy on the top liquid crypto assets using an EMA-21 / EMA-63 crossover for entry/exit signals. Position sizes are scaled inversely to each asset's 14-day ATR, allocating more capital to lower-volatility assets for improved risk-adjusted returns. Benchmarked against a passive BTC hold.

| Metric | Value |
|--------|-------|
| Total Return | +693% |
| CAGR | 64.01% |
| Alpha vs. BTC | 36.92% |
| Signal | EMA 21 / 63 crossover + inverse-ATR sizing |

---

#### Backtest Results Summary

| Strategy | Total Return | CAGR | Alpha | Benchmark |
|---|---|---|---|---|
| US Equities Momentum (SMA 50/200) | +208% | 19.96% | 4.57% | S&P 500 |
| ETF Momentum Rotation (12-1) | +175% | 21.71% | 6.92% | ETF basket |
| Crypto Momentum (EMA 21/63 + ATR) | +693% | 64.01% | 36.92% | BTC hold |

---

#### Tech Stack

```
Runtime        Python 3.13
Backtesting    kaxanuk-backtest-engine
Data           kaxanuk.data_curator
Quant          QuantLib
Data wrangling pandas
```

---

#### How to Run

**Prerequisites**

```bash
pip install kaxanuk-backtest-engine kaxanuk.data_curator QuantLib pandas
```

**US Equities Momentum**

```bash
python strategies/us_equities_momentum.py
```

Runs a daily SMA-50/200 crossover backtest on the configured S&P 500 universe. Results and equity curve are written to `output/us_equities/`.

**ETF Momentum Rotation**

```bash
python strategies/etf_momentum_rotation.py
```

Runs a monthly 12-1 momentum ranking and rotation on the ETF universe. Results written to `output/etf_rotation/`.

**Crypto Momentum**

```bash
python strategies/crypto_momentum.py
```

Runs a daily EMA-21/63 crossover with inverse-ATR position sizing on the crypto universe. Results written to `output/crypto_momentum/`.

---

#### Authors

| Name | Role |
|------|------|
| Néstor Hernández Vázquez | Team leader · Strategy design · Implementation |

`Python` `kaxanuk-backtest-engine` `kaxanuk.data_curator` `QuantLib` `pandas`

---

### [Arrowport](https://github.com/nestor-hdz/arrowport) — AI-Powered Chrome Extension
Chrome extension (Manifest V3) that uses Claude's API to extract 3–5 structured analytical insights from any selected web text. Features a persistent history vault ("La Bóveda"), floating activation button, and drag/resize modal. Published on the Chrome Web Store.
`JavaScript` `Chrome Extensions API` `Anthropic Claude API`

---

### [S&P 500 Portfolio Optimizer](https://github.com/nestor-hdz/S-P-500-Portfolio-Optimizer) — Quantitative Finance
Python tool implementing Modern Portfolio Theory: log returns, 10,000-portfolio Monte Carlo simulation, efficient frontier mapping, maximum Sharpe ratio identification, and max drawdown backtesting. Based on Markowitz (1952).
`Python` `pandas` `NumPy` `SciPy` `matplotlib`

---

### [Momentum-Based Crypto Trading Bot](https://github.com/nestor-hdz/Momentum-based-crypto-trading-bot) — Algorithmic Trading
Modular Python system for systematic momentum trading on Binance Spot. Six-module architecture: config, indicators (EMA 9/21, RSI 14, ADX, ATR), signals, risk manager (Kelly sizing + 15% drawdown kill switch), backtesting engine, and execution layer.
`Python` `pandas` `NumPy` `Binance API`

---

### [Pairs Trading — Statistical Arbitrage](https://github.com/nestor-hdz/pairs-trading) — Quantitative Finance
Statistical arbitrage system for the PEP/KO pair. Computes 20-day rolling Z-score on price ratio, generates entry/exit signals (|Z|≥2.0 / |Z|≤0.5), and displays a rich terminal dashboard with ASCII Z-score gauge. Dockerized.
`Python` `pandas` `yfinance` `rich` `Docker`

---

### [Cerebro](https://github.com/nestor-hdz/cerebro) — Multi-Agent AI Orchestration
Production-grade collaborative AI system where Claude, GPT-4, Perplexity, and Gemini operate as a coordinated swarm. Four-phase pipeline: Briefing → Round Table (role self-allocation) → Synergy Execution → Synthesis with adversarial review. Pydantic models, Redis/in-memory state swap, structured JSON output.
`Python` `asyncio` `Pydantic` `Anthropic SDK` `OpenAI SDK`

---

### [Heist Director](https://github.com/nestor-hdz/heist-director) — AI Narrative Game
Full-stack game where players write a heist plan and Claude generates a cinematic noir film reveal. Features character creation, mission briefing system, Supabase Auth, and a noir film-poster UI aesthetic.
`Next.js` `TypeScript` `Supabase` `Anthropic Claude API` `Tailwind CSS`

---

### [Mi Álbum Mundial 2026](https://github.com/nestor-hdz/mi-album-mundial) — Full-Stack Web App
Web app for tracking FIFA World Cup 2026 sticker albums. OCR scanner (Tesseract.js) reads sticker codes from camera, trade system for duplicates, 994 stickers across 48 countries. Live on Vercel.
`Next.js` `TypeScript` `Supabase` `Tesseract.js` `Tailwind CSS`

---

## Stack

```
Languages   Python · TypeScript · JavaScript · C · MATLAB · SQL
Finance     Portfolio optimization · Monte Carlo · Backtesting
            Statistical arbitrage · Momentum strategies
            Stochastic processes · Probability theory
AI/ML       Anthropic Claude API · OpenAI SDK · Multi-agent orchestration
            OCR (Tesseract.js) · Prompt engineering
Web         Next.js · React · Supabase · Tailwind CSS · Chrome Extensions MV3
Tools       Git · Docker · Vercel · Binance API · yfinance · pandas · NumPy
```

---

## Background

- **ESFM-IPN** — Licenciatura en Física y Matemáticas (2022–2027)
- **MFAI26 Workshop — UNAM** (April 2026) — MIT, Stanford, Berkeley, CMU, EPFL researchers
- **Research — Dr. Carlos Hernández Castellanos, IIMAS-UNAM** — Multi-Objective Reinforcement Learning
- **KaxaNuk Quantitative Finance Research Challenge** (June 2026) — Team leader

---

[![LinkedIn](https://img.shields.io/badge/LinkedIn-nestor--hernandez--vazquez-0077B5?style=flat&logo=linkedin)](https://linkedin.com/in/nestor-hernandez-vazquez)
