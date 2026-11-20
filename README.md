# 📈 Grimes Method Backtest Lab

A one-file, self-contained backtest laboratory for **Adam Grimes' "The Art and Science of Technical Analysis"** (2012) — with rigorous no-lookahead verification, multi-market universes, and built-

![image]([/results/US_15m_20261120_234702/equity.png) 
---

## What This Is

`grimes_lab.py` is a complete backtesting environment in a single Python file. It implements Grimes' core setups — **pullbacks** in trending markets and **failure tests** — across ~126 global markets (US, Europe, Asia, commodities, FX, crypto), with intraday support via yfinance or Alpaca.

The emphasis is on **honest measurement**: the self-test proves there's no lookahead bias, gaps are filled realistically, stops beat targets when ambiguous, and the portfolio only sizes with realized equity. The included checklist tells you whether the edge is *universal* or just hand-picked luck.

---

## Features

- **Two setups from the book**
  - *Pullback*: strong leg → quiet 25–75% retracement → break of signal bar high
  - *Failure test*: probe below a 20-bar low, close back above (spring); mirrored for highs
- **No-lookahead engine** — signals use closed bars only; verified by unit, symmetry, causality, and sizing tests
- **~126 markets** in 5 universes (US ETFs, world indices, international large caps, US stocks, commodities/FX/crypto)
- **Random-entry baseline** — same exits, trend-aligned random entries; tells you if the entry adds anything
- **Out-of-sample split** and **parameter robustness grid** for checking curve-fit
- **Random-subset test** — if the edge is universal, most random subsets should be positive
- **Intraday** support (5m/15m/60m) with accumulating cache (yfinance) or Alpaca
- **Full terminal UI** with settings, report viewer, and colour output
- **Outputs per run**: `trades.csv`, `per_market.csv`, `equity.png`, `scoreboard.png`, `report.txt`

---

## Installation

```bash
pip install yfinance pandas numpy scipy matplotlib
# Optional — for Alpaca intraday:
pip install alpaca-py
```

Python 3.8+ recommended.

---

## Quick Start

```bash
# Full interactive terminal menu
python grimes_lab.py

# Verify no-lookahead / fill logic / symmetry / sizing
python grimes_lab.py --selftest

# Synthetic-data smoke test (no internet needed)
python grimes_lab.py --demo
```

Inside the menu:

| Option | Action |
|--------|--------|
| **1** | Self-test — proves the engine is honest |
| **2** | Download/refresh daily data for a market set |
| **3** | Backtest a market set |
| **4** | Backtest + random-entry baseline |
| **5** | **Universality test** — all markets worldwide (main honesty check) |
| **6** | Random-subset test — is it hand-picked luck? |
| **7** | Parameter robustness grid — is it curve-fit? |
| **8** | Intraday with yfinance |
| **9** | Intraday with Alpaca (needs API keys) |
| **10** | Settings — costs, risk, shorts, dates, setups |
| **11** | View a previous report |
| **12** | Help — what do the numbers mean? |

---

## No-Lookahead Rules (verified by `--selftest`)

- Signal on bar `t` uses data up to the **close** of bar `t` only; the order is live from bar `t+1`
- Buy-stops that gap through the trigger fill at the **open**
- Failure-test entry = **next bar's open**
- Stop and target inside the same bar → **stop is assumed first**
- Weekly trend uses **completed weeks only**
- Position size uses **realized equity known at entry only**

The self-test also verifies long-vs-mirrored-short symmetry and that high-priced instruments (e.g. index level ~30,000) are properly sized, not skipped.

---

## Reading the Results

| Metric | Meaning |
|--------|---------|
| **R** | Profit ÷ initial risk. −1R = stopped out, +2R = won twice what you risked |
| **avg R** | Expectancy per trade. Above 0 *after costs* is the minimum for an edge |
| **t-stat** | avg R ÷ its standard error. ≥3 is "strong" (we test many things, so be strict) |
| **win %** | Meaningless alone — 35% win rate can be great with 3R winners |
| **PF** | Profit factor = gross wins ÷ gross losses |
| **OOS** | Out-of-sample trades. Great IS + flat OOS = fitted/luck |
| **baseline** | Random entries with the same exits. If you don't beat it, the entry adds nothing |

### The Evidence Checklist

Every full study ends with a PASS/FAIL list:

- Enough trades (N ≥ 300)
- Positive expectancy after costs
- Statistically strong (t ≥ 3)
- Works across markets (≥60% positive, sign-test p < 0.10)
- Works across regions/asset classes
- Holds out-of-sample
- Beats the random-entry baseline

**All pass** → candidate for paper-trading. **Mostly pass** → investigate the weak spot. **Otherwise** → NOT demonstrated.

> ⚠️ p-values assume independent trades; overlapping trades in correlated markets make them optimistic. Treat them as a guide, not a verdict.

---

## The Setups

### Pullback
In a trending market (20-bar majority above/below the 20 EMA, weekly trend agrees) after a strong leg (≥4 ATR), buy the first quiet 25–75% retracement when price breaks the signal bar's high.

### Failure Test
Price probes below a 20-bar low and closes back above it (spring). Mirrored for highs.

### Exits (both setups)
- Stop ≥ 2 ATR from entry
- Half off at +1R; stop to breakeven
- Rest to measured move / 3R
- 15-bar time stop

---

## Data Sources

| Source | Coverage | Notes |
|--------|----------|-------|
| **yfinance (daily)** | Full history, ~126 symbols | Cached in `cache_daily/`, refreshed if >20h old |
| **yfinance (intraday)** | Last 60d (5m/15m/30m) or 730d (60m) | Merges into `cache_intraday/` — history accumulates if re-run weekly |
| **Alpaca** | Years of intraday (needs keys) | Set `APCA_API_KEY_ID` / `APCA_API_SECRET_KEY` in env, `pip install alpaca-py` |

For multi-year intraday history, use Alpaca. yfinance only returns the last ~60 days of sub-hourly bars per request.

---

## Universes

```
US ETFs                    ~23 markets   (SPY, QQQ, sectors, intl, bonds/gold)
World stock indices        ~22 markets   (US, Europe, Asia-Pacific, Americas)
International large caps   ~40 markets   (DE, FR, UK, CH/NL, JP, HK/CN, KR/IN/AU, CA)
US large-cap stocks        ~20 markets
Commodities / FX / crypto  ~18 markets   (futures, FX pairs, BTC/ETH)
```

Or choose: **ALL** (~126 markets), a **random sample** (removes hand-picking), **custom tickers**, or **synthetic demo data**.

---

## Honesty Rules Built In

- Signals use only closed bars; fills happen on **later** bars
- Gaps fill at the open
- Stop beats target inside a bar
- Results on synthetic data mean **nothing** about markets — they only test the machinery
- Past performance, even honestly measured, is not a promise

**Paper-trade before real money.**

---

## Output Files

Each run creates `results/<name>_<timestamp>/`:

| File | Contents |
|------|----------|
| `trades.csv` | Every trade: symbol, setup, side, times, entry, risk, R, bars, reason, T1 hit |
| `per_market.csv` | N, win rate, avg R, t-stat per market |
| `equity.png` | Equity curve, drawdown, cumulative R by setup, R distribution |
| `scoreboard.png` | Avg R per market with error bars, trade-weighted avg R by group |
| `report.txt` | Full text report including the evidence checklist |

---

## Example

```
$ python grimes_lab.py --demo

================ demo synthetic  (14 markets, 12.3s) ================
setups: pullback, failtest | shorts: True | slippage 5.0 bps/side | risk 1.00%/trade | weekly filter: True

POOLED (R = profit / initial risk, after slippage):
N            412.000
win            0.381
avgR          -0.042
...
---------- EVIDENCE CHECKLIST (is it universal, or hand-picked luck?) ----------
  [FAIL] enough trades (N=412, want >= 300)        ← synthetic data, no real edge
  [FAIL] positive expectancy after costs (avg R = -0.042)
  ...
  => NOT demonstrated: the evidence does not support a universal, automatable edge for these rules.
```

*Synthetic data contains no edge — that's the point. It tests the machinery, not the market.*

---

## Project Structure

```
grimes_lab.py            ← the entire lab (single file)
settings.json            ← persisted settings (auto-created)
cache_daily/             ← cached daily CSVs per symbol
cache_intraday/          ← cached intraday PKLs per symbol+interval
results/
  └── <name>_<time>/
      ├── trades.csv
      ├── per_market.csv
      ├── equity.png
      ├── scoreboard.png
      └── report.txt
```

---

## Requirements

- Python 3.8+
- `pandas`, `numpy`, `scipy`, `matplotlib`
- `yfinance` (daily + yfinance intraday)
- `alpaca-py` (optional, for Alpaca intraday)

---

## Disclaimer

This is a **research tool**, not investment advice. Backtests are subject to survivorship bias, data errors, and regime change. Even an honest backtest with no lookahead and out-of-sample validation is not a prediction. Use it to learn, not to trade. You are responsible for your own decisions.

---

## Credits

Methodology from **Adam Grimes, *The Art and Science of Technical Analysis: Market Structure, Price Action, and Trading Strategies* (Wiley, 2012)**.

Engine, universes, and honesty checks by the `grimes_lab.py` authors. This README is part of the repository; edit freely.

---

## License

MIT — see `LICENSE`. Use, modify, and share; no warranty.
