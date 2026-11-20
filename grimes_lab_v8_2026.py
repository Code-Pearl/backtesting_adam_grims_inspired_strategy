#!/usr/bin/env python3
"""
grimes_lab.py  --  ONE-FILE backtest lab for Adam Grimes' "The Art and Science of Technical Analysis" (2012)

Start it:        python grimes_lab.py            (opens the full terminal menu)
Other options:   python grimes_lab.py --selftest (prove no-lookahead / fill logic / symmetry / sizing)
                 python grimes_lab.py --demo     (synthetic-data smoke test, no internet needed)
Install:         pip install yfinance pandas numpy scipy matplotlib        (+ alpaca-py for the Alpaca menu item)

Contents of this file
  PART 1  engine      indicators, signals (pullback + failure test), fill simulator, portfolio, statistics, verdict
  PART 2  data        yfinance daily loader with cache, synthetic data, self-tests
  PART 3  universes   ~126 markets worldwide in groups (US, Europe, Asia, commodities/FX/crypto ...)
  PART 4  intraday    yfinance (accumulating cache) and Alpaca sources, session-aware preparation
  PART 5  terminal UI menu, settings, report viewer, help

NO-LOOKAHEAD RULES (verified by the self-test)
  * Signal on bar t uses data up to the CLOSE of bar t only; the order is live from bar t+1.
  * Buy-stops that gap through the trigger fill at the OPEN. Failure-test entry = NEXT bar's open.
  * Stop and target inside the same bar -> the STOP is assumed first.
  * Weekly trend uses completed weeks only. Position size uses realized equity known at entry only.
Outputs per run: results/<name>_<time>/ trades.csv, per_market.csv, equity.png, scoreboard.png, report.txt
"""
from __future__ import annotations
import contextlib, io, json, math, os, random, sys, time
from dataclasses import dataclass, replace
import numpy as np
import pandas as pd

LAST_RUN: dict = {}


# =====================================================================================================================
#  PART 1 -- ENGINE + DATA + TESTS
# =====================================================================================================================
TICK = 0.01
WARM = 130


# ----------------------------------------------------------------------------- config
@dataclass
class Cfg:
    ema_len: int = 20
    atr_len: int = 20
    kelt_mult: float = 2.25
    trend_lookback: int = 20
    trend_min_frac: float = 0.70
    use_htf: bool = True
    htf_ema: int = 20
    pb_window: int = 15
    pb_min_bars: int = 2
    pb_max_bars: int = 12
    leg_lookback: int = 20
    leg_min_atr: float = 4.0
    retr_min: float = 0.25
    retr_max: float = 0.75
    max_dist_ema_atr: float = 1.0
    macd_lookback: int = 60
    ft_lookback: int = 20
    ft_min_age: int = 3
    ft_max_exc_atr: float = 1.5
    ft_need_keltner: bool = False
    ft_min_risk_atr: float = 0.75
    stop_min_atr: float = 2.0
    stop_max_atr: float = 4.0
    stop_buffer_atr: float = 0.25
    t1_r: float = 1.0
    t1_frac: float = 0.5
    t2_r: float = 3.0
    max_hold: int = 15
    trigger_valid_bars: int = 1
    slippage_bps: float = 5.0
    risk_frac: float = 0.01
    max_notional_frac: float = 0.20
    max_gross: float = 1.0
    max_positions: int = 8
    start_equity: float = 100_000.0
    allow_short: bool = True
    flatten_eod: bool = False     # intraday: flat at the last bar of each session
    same_session: bool = False    # intraday: whole setup must lie inside the current session


# ----------------------------------------------------------------------------- indicators
def add_indicators(df: pd.DataFrame, cfg: Cfg) -> pd.DataFrame:
    """Causal: every value at row t uses rows <= t only."""
    d = df.copy()
    pc = d["close"].shift(1)
    tr = pd.concat([d["high"] - d["low"], (d["high"] - pc).abs(), (d["low"] - pc).abs()], axis=1).max(axis=1)
    d["atr"] = tr.ewm(alpha=1 / cfg.atr_len, adjust=False, min_periods=cfg.atr_len).mean()
    d["ema"] = d["close"].ewm(span=cfg.ema_len, adjust=False, min_periods=cfg.ema_len).mean()
    d["macd"] = d["close"].rolling(3).mean() - d["close"].rolling(10).mean()   # Grimes' modified MACD fast line
    d["above"] = (d["close"] > d["ema"]).astype(float).where(d["ema"].notna()).rolling(cfg.trend_lookback).mean()
    return d


def weekly_trend(d: pd.DataFrame, cfg: Cfg) -> pd.Series:
    """1.0 if the latest COMPLETED weekly close > weekly EMA. Weekly bars are labelled by Friday; a day takes the
    latest label <= itself, so Mon-Thu see last week's finished bar and Friday sees its own close (known at close)."""
    w = d["close"].resample("W-FRI").last().dropna()
    wema = w.ewm(span=cfg.htf_ema, adjust=False, min_periods=cfg.htf_ema).mean()
    up = (w > wema).astype(float).where(wema.notna())
    return up.reindex(d.index, method="ffill")


def prepare(raw: pd.DataFrame, cfg: Cfg) -> pd.DataFrame:
    d = add_indicators(raw, cfg)
    d["htf"] = weekly_trend(d, cfg)
    return d


# ----------------------------------------------------------------------------- signals
def _view(d: pd.DataFrame, side: int) -> dict:
    """Arrays in 'long space'. Shorts are mirrored (price x -1, high<->low) so ONE block of logic serves both sides;
    R-multiples are identical in both spaces (verified by the symmetry test)."""
    f = lambda k: d[k].values.astype(float)
    if side == 1:
        return dict(o=f("open"), h=f("high"), l=f("low"), c=f("close"), ema=f("ema"),
                    macd=f("macd"), above=f("above"), htf=f("htf"), atr=f("atr"))
    return dict(o=-f("open"), h=-f("low"), l=-f("high"), c=-f("close"), ema=-f("ema"),
                macd=-f("macd"), above=1 - f("above"), htf=1 - f("htf"), atr=f("atr"))


def pullback_signals(v, cfg, side, ok, sess):
    o, h, l, c, ema, macd, above, htf, atr = (v[k] for k in
                                              ("o", "h", "l", "c", "ema", "macd", "above", "htf", "atr"))
    n, out = len(c), []
    for t in range(WARM, n - 1):
        if not ok[t]:
            continue
        a = atr[t]
        if not (a > 0):
            continue
        if not (above[t] >= cfg.trend_min_frac):                      # 1 trending on this time frame
            continue
        if cfg.use_htf and not (htf[t] == 1.0):                       # 2 higher time frame agrees
            continue
        w0 = max(0, t - cfg.pb_window + 1)
        hi = w0 + int(np.argmax(h[w0:t + 1]))                         # 3 top of the setup leg
        nb = t - hi
        if nb < cfg.pb_min_bars or nb > cfg.pb_max_bars:
            continue
        s0 = max(0, hi - cfg.leg_lookback)
        lo = s0 + int(np.argmin(l[s0:hi + 1]))                        #   bottom of the setup leg
        if hi - lo < 2:
            continue
        if sess is not None and sess[lo] != sess[t]:
            continue
        leg = h[hi] - l[lo]
        if leg < cfg.leg_min_atr * a:                                 # 4 the leg is a real impulse
            continue
        pb = l[hi + 1:t + 1].min()
        if not (cfg.retr_min <= (h[hi] - pb) / leg <= cfg.retr_max):  # 5 depth 25-75 %
            continue
        if (c[t] - ema[t]) > cfg.max_dist_ema_atr * a:                # 6 not stretched above the average
            continue
        m0 = max(0, lo - cfg.macd_lookback)                           # 7 no momentum divergence
        if lo - m0 < 10:
            continue
        if not (np.nanmax(macd[lo:hi + 1]) >= np.nanmax(macd[m0:lo])):
            continue
        if np.mean(h[hi + 1:t + 1] - l[hi + 1:t + 1]) > np.mean(h[lo + 1:hi + 1] - l[lo + 1:hi + 1]):
            continue                                                  # 8 pullback quieter than the leg
        out.append(dict(t=t, side=side, kind="pullback", entry="stop", trig=h[t] + TICK,
                        pb_ext=pb, leg=leg, atr=a))
    return out


def failtest_signals(v, cfg, side, ok, sess):
    o, h, l, c, ema, atr = (v[k] for k in ("o", "h", "l", "c", "ema", "atr"))
    n, out = len(c), []
    for t in range(WARM, n - 1):
        if not ok[t]:
            continue
        a = atr[t]
        if not (a > 0):
            continue
        w0 = t - cfg.ft_lookback
        if sess is not None and sess[w0] != sess[t]:
            continue
        seg = l[w0:t]
        sup = seg.min()
        si = w0 + int(np.argmin(seg))
        if t - si < cfg.ft_min_age:
            continue
        if not (l[t] < sup and c[t] > sup):
            continue
        if (sup - l[t]) > cfg.ft_max_exc_atr * a:
            continue
        if (c[t] - l[t]) < 0.5 * (h[t] - l[t]):
            continue
        if cfg.ft_need_keltner and not (l[t] <= ema[t] - cfg.kelt_mult * a):
            continue
        out.append(dict(t=t, side=side, kind="failtest", entry="open", trig=None, pb_ext=l[t], leg=0.0, atr=a))
    return out


def generate_signals(d: pd.DataFrame, cfg: Cfg, kinds=("pullback", "failtest")) -> list:
    ok = d["ok_entry"].values.astype(bool) if "ok_entry" in d else np.ones(len(d), bool)
    sess = d["sess"].values if ("sess" in d and cfg.same_session) else None
    sigs = []
    for side in ([1, -1] if cfg.allow_short else [1]):
        v = _view(d, side)
        if "pullback" in kinds:
            sigs += pullback_signals(v, cfg, side, ok, sess)
        if "failtest" in kinds:
            sigs += failtest_signals(v, cfg, side, ok, sess)
    return sigs


def random_signals(d, cfg, rng, p):
    """Baseline: random entries in the direction of the trend filters, same exits as the strategy."""
    ok = d["ok_entry"].values.astype(bool) if "ok_entry" in d else np.ones(len(d), bool)
    out = []
    for side in ([1, -1] if cfg.allow_short else [1]):
        v = _view(d, side)
        for t in range(WARM, len(d) - 1):
            if ok[t] and v["above"][t] >= cfg.trend_min_frac and (not cfg.use_htf or v["htf"][t] == 1.0) \
                    and v["atr"][t] > 0 and rng.random() < p:
                out.append(dict(t=t, side=side, kind="random", entry="open", trig=None,
                                pb_ext=v["l"][t], leg=0.0, atr=v["atr"][t]))
    return out


# ----------------------------------------------------------------------------- simulator
def simulate(d: pd.DataFrame, sigs: list, cfg: Cfg, symbol: str = "") -> list:
    n, idx = len(d), d.index
    views = {1: _view(d, 1), -1: _view(d, -1)}
    sess = d["sess"].values if "sess" in d else None
    bps = cfg.slippage_bps / 1e4
    trades, last_exit = [], -1
    for s in sorted(sigs, key=lambda x: x["t"]):
        t, side, a = s["t"], s["side"], s["atr"]
        if last_exit > t:                      # still in a trade when this signal printed
            continue
        v = views[side]
        o, h, l, c = v["o"], v["h"], v["l"], v["c"]
        j = e = None
        if s["entry"] == "stop":
            for k in range(t + 1, min(t + 1 + cfg.trigger_valid_bars, n)):
                if o[k] >= s["trig"]:
                    j, e = k, o[k]; break               # gapped through trigger: fill at open
                if h[k] >= s["trig"]:
                    j, e = k, s["trig"]; break
        elif t + 1 < n:
            j, e = t + 1, o[t + 1]
        if j is None:
            continue
        e = e + abs(e) * bps
        if s["kind"] == "pullback":
            stop = min(s["pb_ext"] - cfg.stop_buffer_atr * a, e - cfg.stop_min_atr * a)
            t2 = max(s["pb_ext"] + s["leg"], e + 2 * (e - stop))          # measured-move objective
        elif s["kind"] == "failtest":
            stop = s["pb_ext"] - 0.1 * a
            if e - stop < cfg.ft_min_risk_atr * a:
                stop = e - cfg.ft_min_risk_atr * a
            t2 = None
        else:
            stop, t2 = e - cfg.stop_min_atr * a, None
        risk = e - stop
        if risk <= 0 or risk > cfg.stop_max_atr * a:
            continue
        t1 = e + cfg.t1_r * risk
        if t2 is None:
            t2 = e + cfg.t2_r * risk
        rem, R, t1_done, reason, xk = 1.0, 0.0, False, "time", j
        for k in range(j, n):
            xk = k
            if k > j and o[k] <= stop:
                px = o[k] - abs(o[k]) * bps
                R += rem * (px - e) / risk; rem = 0.0; reason = "stop_gap"; break
            if l[k] <= stop:                                              # stop first when ambiguous
                px = stop - abs(stop) * bps
                R += rem * (px - e) / risk; rem = 0.0; reason = "BE_stop" if t1_done else "stop"; break
            if not t1_done and h[k] >= t1:
                fill = max(t1, o[k]) if k > j else t1
                R += cfg.t1_frac * (fill - e) / risk; rem -= cfg.t1_frac
                t1_done = True; stop = max(stop, e)
            if t1_done and h[k] >= t2:
                fill = max(t2, o[k]) if k > j else t2
                R += rem * (fill - e) / risk; rem = 0.0; reason = "target2"; break
            eod = cfg.flatten_eod and sess is not None and (k == n - 1 or sess[k + 1] != sess[k])
            if k - j >= cfg.max_hold or k == n - 1 or eod:
                px = c[k] - abs(c[k]) * bps
                R += rem * (px - e) / risk; rem = 0.0; reason = "eod" if eod else "time"; break
        last_exit = xk
        trades.append(dict(symbol=symbol, kind=s["kind"], side="long" if side == 1 else "short",
                           signal_time=idx[t], entry_time=idx[j], exit_time=idx[xk],
                           entry=abs(e), risk=risk, R=R, bars=xk - j, reason=reason, t1_hit=t1_done))
    return trades


def trades_for(prepared: dict, cfg: Cfg, kinds=("pullback", "failtest")) -> pd.DataFrame:
    out = []
    for s, d in prepared.items():
        out += simulate(d, generate_signals(d, cfg, kinds), cfg, s)
    tr = pd.DataFrame(out)
    for c in ("signal_time", "entry_time", "exit_time"):      # one clock for all markets (UTC, naive)
        if len(tr):
            tr[c] = pd.to_datetime(tr[c], utc=True).dt.tz_localize(None)
    return tr


# ----------------------------------------------------------------------------- portfolio + statistics
def portfolio(trades: pd.DataFrame, cfg: Cfg):
    """Chronological replay, fixed-fractional risk on REALIZED equity. Sizing is currency-free
    (notional = risk$ * entry/risk) so index levels like 30,000 and FX rates like 1.08 are treated alike."""
    tr = trades.sort_values("entry_time").reset_index(drop=True)
    eq, open_pos, closed, skipped = cfg.start_equity, [], [], 0
    for t in tr.itertuples():
        still = []
        for p in open_pos:
            if p["exit"] < t.entry_time:
                eq += p["pnl"]; closed.append(p)
            else:
                still.append(p)
        open_pos = still
        if len(open_pos) >= cfg.max_positions:
            skipped += 1; continue
        gross = sum(p["notional"] for p in open_pos)
        notional = min(cfg.risk_frac * eq * t.entry / t.risk, cfg.max_notional_frac * eq, cfg.max_gross * eq - gross)
        if notional <= 1e-9:
            skipped += 1; continue
        open_pos.append(dict(exit=t.exit_time, notional=notional, pnl=t.R * notional * t.risk / t.entry))
    closed += open_pos
    closed.sort(key=lambda p: p["exit"])
    if not closed:
        return pd.Series(dtype=float), skipped
    curve = pd.Series(cfg.start_equity + np.cumsum([p["pnl"] for p in closed]), index=[p["exit"] for p in closed])
    return curve, skipped


def r_stats(R) -> dict:
    R = np.asarray(R, float)
    n = len(R)
    if n < 3:
        return dict(N=n)
    from scipy import stats
    mean, sd = R.mean(), R.std(ddof=1)
    tstat = mean / (sd / math.sqrt(n)) if sd > 0 else float("nan")
    w, lo = R[R > 0], R[R < 0]
    return dict(N=n, win=float((R > 0).mean()), avgR=float(mean), sd=float(sd),
                avg_win=float(w.mean()) if len(w) else 0.0, avg_loss=float(lo.mean()) if len(lo) else 0.0,
                PF=float(w.sum() / -lo.sum()) if len(lo) else float("inf"),
                t=float(tstat), p_one_tailed=float(stats.t.sf(tstat, n - 1)))


def stat_table(df: pd.DataFrame, by: str) -> pd.DataFrame:
    return pd.DataFrame({k: r_stats(g["R"]) for k, g in df.groupby(by)}).T.round(3)


def per_market(tr: pd.DataFrame, symbols, groups: dict) -> pd.DataFrame:
    rows = []
    for s in symbols:
        g = tr[tr["symbol"] == s]
        st = r_stats(g["R"]) if len(g) else dict(N=0)
        st.update(symbol=s, group=groups.get(s, ""))
        rows.append(st)
    return pd.DataFrame(rows).set_index("symbol")


def random_baseline(prepared: dict, cfg: Cfg, sims=30, p=0.03, seed=7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    res = []
    for _ in range(sims):
        rs = []
        for s, d in prepared.items():
            rs += [t["R"] for t in simulate(d, random_signals(d, cfg, rng, p), cfg, s)]
        res.append(np.mean(rs) if rs else np.nan)
    return np.array(res)


def param_grid(prepared, cfg, kinds, n1, v1, n2, v2):
    """avg R and N for a small grid of neighbouring parameter values. A real edge survives neighbours."""
    avg = pd.DataFrame(index=v1, columns=v2, dtype=float)
    cnt = pd.DataFrame(index=v1, columns=v2, dtype=float)
    for a in v1:
        for b in v2:
            tr = trades_for(prepared, replace(cfg, **{n1: a, n2: b}), kinds)
            avg.loc[a, b] = tr["R"].mean() if len(tr) else np.nan
            cnt.loc[a, b] = len(tr)
    avg.index.name, avg.columns.name = n1, n2
    return avg.round(3), cnt.astype(int)


# ----------------------------------------------------------------------------- verdict
def verdict(tr, pm, oos_stats, baseline_mean_dist, pb_mean, group_avg):
    from scipy import stats
    st = r_stats(tr["R"])
    items = []
    items.append((st["N"] >= 300, f"enough trades (N={st['N']}, want >= 300)"))
    items.append((st.get("avgR", -1) > 0, f"positive expectancy after costs (avg R = {st.get('avgR', float('nan')):+.3f})"))
    items.append((st.get("t", 0) >= 3, f"statistically strong (t = {st.get('t', float('nan')):.2f}, want >= 3 because many things were tried)"))
    good = pm[pm["N"] >= 10]
    if len(good) >= 6:
        k = int((good["avgR"] > 0).sum()); m = len(good)
        pv = stats.binomtest(k, m, 0.5, alternative="greater").pvalue
        items.append((k / m >= 0.6 and pv < 0.10, f"works across markets ({k}/{m} markets positive, sign-test p = {pv:.3f})"))
    else:
        items.append((False, "works across markets (too few markets with >= 10 trades)"))
    if group_avg is not None and len(group_avg) >= 2:
        pos = int((group_avg > 0).sum())
        items.append((pos / len(group_avg) >= 2 / 3, f"works across regions/asset classes ({pos}/{len(group_avg)} groups positive)"))
    if oos_stats is not None and oos_stats.get("N", 0) >= 30:
        items.append((oos_stats["avgR"] > 0, f"holds out-of-sample (avg R = {oos_stats['avgR']:+.3f}, N={oos_stats['N']})"))
    if baseline_mean_dist is not None and pb_mean is not None:
        hi = float(np.nanpercentile(baseline_mean_dist, 95))
        items.append((pb_mean > hi, f"pullback entries beat random entries with identical exits ({pb_mean:+.3f} vs 95th pct of random {hi:+.3f})"))
    passed = sum(1 for ok, _ in items if ok)
    lines = [("  [PASS] " if ok else "  [FAIL] ") + txt for ok, txt in items]
    if passed == len(items):
        summ = "All checks pass -> candidate worth paper-trading, but still verify on data you have never looked at."
    elif passed >= len(items) - 1:
        summ = "Mostly consistent, one weak spot -> investigate the failing check before trusting it."
    else:
        summ = "NOT demonstrated: the evidence does not support a universal, automatable edge for these rules."
    return lines, summ


# ----------------------------------------------------------------------------- plots
def save_equity_png(tr, curve, cfg, path, title):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 2, figsize=(15, 9))
    ax[0, 0].step(curve.index, curve.values, where="post", lw=1.4)
    ax[0, 0].axhline(cfg.start_equity, color="gray", lw=.8, ls="--")
    ax[0, 0].set_title(f"Equity (realized, {cfg.risk_frac:.1%} risk/trade) - {title}")
    dd = curve / curve.cummax() - 1
    ax[1, 0].fill_between(curve.index, dd.values * 100, 0, step="post", color="crimson", alpha=.5)
    ax[1, 0].set_title("Drawdown % (realized trades only)")
    t2 = tr.sort_values("exit_time")
    for k, g in t2.groupby("kind"):
        ax[0, 1].step(g["exit_time"], g["R"].cumsum(), where="post", label=f"{k} (N={len(g)})")
    ax[0, 1].legend(); ax[0, 1].set_title("Cumulative R by setup (position-size independent)")
    ax[1, 1].hist(tr["R"].clip(-3, 5), bins=40, color="steelblue")
    ax[1, 1].axvline(tr["R"].mean(), color="k", ls="--", label=f"mean {tr['R'].mean():+.3f}R")
    ax[1, 1].legend(); ax[1, 1].set_title("Distribution of trade results (R, clipped -3..5)")
    for a in ax.ravel():
        a.grid(alpha=.25)
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def save_scoreboard_png(pm, tr, path, title):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = pm[pm["N"] >= 3].copy()
    if d.empty:
        return
    d["se"] = d["sd"] / np.sqrt(d["N"])
    d = d.sort_values("avgR")
    groups = list(dict.fromkeys(d["group"]))
    cmap = plt.get_cmap("tab20")
    col = {g: cmap(i % 20) for i, g in enumerate(groups)}
    fig, ax = plt.subplots(1, 2, figsize=(15, max(6, .28 * len(d))), gridspec_kw={"width_ratios": [3, 1.3]})
    ax[0].barh(d.index, d["avgR"], xerr=d["se"], color=[col[g] for g in d["group"]], alpha=.9)
    ax[0].axvline(0, color="k", lw=1)
    ax[0].axvline(tr["R"].mean(), color="gray", ls="--", lw=1, label=f"pooled {tr['R'].mean():+.3f}R")
    ax[0].set_title(f"Average R per market (bars = +/-1 s.e.) - {title}"); ax[0].legend(); ax[0].grid(alpha=.25)
    for g in groups:
        ax[0].barh([], [], color=col[g], label=g)
    ga = d.groupby("group").apply(lambda x: np.average(x["avgR"], weights=x["N"]), include_groups=False).sort_values()
    ax[1].barh(ga.index, ga.values, color=[col[g] for g in ga.index]); ax[1].axvline(0, color="k", lw=1)
    ax[1].set_title("Trade-weighted avg R by group"); ax[1].grid(alpha=.25)
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


# ----------------------------------------------------------------------------- full study (prints + saves everything)
class _Tee:
    def __init__(self, *f): self.f = f
    def write(self, x):
        for f in self.f: f.write(x)
    def flush(self):
        for f in self.f: f.flush()


def study(data: dict, groups: dict, cfg: Cfg, kinds, oos, outdir, name, baseline=False, sims=30, prepared=None) -> pd.DataFrame:
    os.makedirs(outdir, exist_ok=True)
    buf = io.StringIO()
    with contextlib.redirect_stdout(_Tee(sys.stdout, buf)):
        t0 = time.time()
        if prepared is None:
            prepared = {s: prepare(raw, cfg) for s, raw in data.items()}
        tr = trades_for(prepared, cfg, kinds)
        print(f"\n================ {name}  ({len(data)} markets, {time.time() - t0:.1f}s) ================")
        print(f"setups: {', '.join(kinds)} | shorts: {cfg.allow_short} | slippage {cfg.slippage_bps} bps/side | "
              f"risk {cfg.risk_frac:.2%}/trade | weekly filter: {cfg.use_htf}")
        if tr.empty:
            print("no trades"); return tr
        tr["group"] = tr["symbol"].map(groups)
        et = pd.to_datetime(tr["entry_time"])
        print("\nPOOLED (R = profit / initial risk, after slippage):")
        print(pd.Series(r_stats(tr["R"])).round(3).to_string())
        print("\nBy setup:\n", stat_table(tr, "kind").to_string())
        print("\nBy side:\n", stat_table(tr, "side").to_string())
        print("\nBy group:\n", stat_table(tr, "group").to_string())
        span_days = (et.max() - et.min()).days
        tr["period"] = et.dt.year.astype(str) if span_days > 400 else et.dt.strftime("%Y-%m")
        print("\nBy period (N, avg R):\n", tr.groupby("period")["R"].agg(["count", "mean"]).round(3).T.to_string())
        oos_stats = None
        if oos:
            cut = pd.Timestamp(oos)
            a, b = tr[et < cut], tr[et >= cut]
            print(f"\nIN-SAMPLE  (< {oos}): ", {k: round(v, 3) for k, v in r_stats(a["R"]).items()})
            oos_stats = r_stats(b["R"])
            print(f"OUT-OF-SAMPLE (>= {oos}):", {k: round(v, 3) for k, v in oos_stats.items()})
        pm = per_market(tr, list(data.keys()), groups)
        pm_show = pm[pm["N"] > 0][["group", "N", "win", "avgR", "t"]].sort_values("avgR", ascending=False)
        print("\nPER MARKET (best to worst):\n", pm_show.round(3).to_string())
        curve, skipped = portfolio(tr, cfg)
        if len(curve) > 2:
            yrs = max((curve.index[-1] - curve.index[0]).days / 365.25, 1e-9)
            dd = (curve / curve.cummax() - 1).min()
            print(f"\nPORTFOLIO: final ${curve.iloc[-1]:,.0f} (start ${cfg.start_equity:,.0f})  "
                  f"CAGR {(curve.iloc[-1] / cfg.start_equity) ** (1 / yrs) - 1:.2%}  maxDD(realized only) {dd:.1%}  "
                  f"skipped(no capacity) {skipped}")
        bdist = pbm = None
        if baseline:
            bdist = random_baseline(prepared, cfg, sims)
            pb = tr[tr["kind"] == "pullback"]["R"]
            pbm = float(pb.mean()) if len(pb) else None
            print(f"\nRANDOM-ENTRY BASELINE (same exits, trend-aligned random entries), {sims} runs: "
                  f"mean R {np.nanmean(bdist):+.3f} (5th-95th pct {np.nanpercentile(bdist, 5):+.3f} .. {np.nanpercentile(bdist, 95):+.3f})")
        gavg = None
        gg = pm[pm["N"] >= 10].groupby("group").apply(lambda x: np.average(x["avgR"], weights=x["N"]), include_groups=False)
        if len(gg):
            gavg = gg
        lines, summ = verdict(tr, pm, oos_stats, bdist, pbm, gavg)
        print("\n---------- EVIDENCE CHECKLIST (is it universal, or hand-picked luck?) ----------")
        print("\n".join(lines)); print("  =>", summ)
        LAST_RUN.update(name=name, N=int(len(tr)), avgR=float(tr["R"].mean()), outdir=outdir,
                        passes=sum(1 for l in lines if l.startswith("  [PASS]")), total=len(lines))
        print("  Note: p-values assume independent trades; overlapping trades in correlated markets make them optimistic.")
        tr.drop(columns=["period"]).to_csv(os.path.join(outdir, "trades.csv"), index=False)
        pm.to_csv(os.path.join(outdir, "per_market.csv"))
        if len(curve) > 2:
            save_equity_png(tr, curve, cfg, os.path.join(outdir, "equity.png"), name)
        save_scoreboard_png(pm, tr, os.path.join(outdir, "scoreboard.png"), name)
        print(f"\nSaved to {outdir}/ : trades.csv, per_market.csv, equity.png, scoreboard.png, report.txt")
    with open(os.path.join(outdir, "report.txt"), "w") as fh:
        fh.write(buf.getvalue())
    return tr


# ----------------------------------------------------------------------------- data (daily, cached)
def _safe(s): return s.replace("^", "_").replace("=", "_").replace("/", "_")


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df[["open", "high", "low", "close"]].astype(float).dropna()
    df = df[(df > 0).all(axis=1)]
    df["high"] = df[["open", "high", "low", "close"]].max(axis=1)
    df["low"] = df[["open", "high", "low", "close"]].min(axis=1)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df


def load_yf(symbols, start="2000-01-01", cache="cache_daily", max_age_h=20, force=False):
    """Full history cached per symbol (CSV). Re-downloaded when older than max_age_h or force=True.
    If a download fails but a stale cache exists, the stale cache is used (and says so)."""
    import yfinance as yf
    os.makedirs(cache, exist_ok=True)
    out, failed = {}, []
    for i, s in enumerate(symbols, 1):
        fp = os.path.join(cache, _safe(s) + ".csv")
        have = os.path.exists(fp)
        fresh = have and not force and (time.time() - os.path.getmtime(fp)) / 3600 < max_age_h
        df, tag = None, ""
        if fresh:
            df, tag = pd.read_csv(fp, index_col=0, parse_dates=True), "cache"
        else:
            for _ in range(3):
                try:
                    x = yf.download(s, period="max", interval="1d", auto_adjust=True, progress=False, threads=False)
                    if x is not None and not x.empty:
                        if isinstance(x.columns, pd.MultiIndex):
                            x.columns = x.columns.get_level_values(0)
                        x.columns = [str(c).lower() for c in x.columns]
                        x.index = pd.to_datetime(x.index).tz_localize(None)
                        df = _clean(x); df.to_csv(fp); tag = "downloaded"; break
                except Exception as ex:
                    err = str(ex)[:80]
                time.sleep(1.0)
            if df is None and have:
                df, tag = pd.read_csv(fp, index_col=0, parse_dates=True), "STALE cache (download failed)"
        if df is None or len(df) < WARM + 50:
            failed.append(s); print(f"  [{i}/{len(symbols)}] {s:<12} FAILED"); continue
        df.index = pd.to_datetime(df.index).tz_localize(None)
        df = df[df.index >= pd.Timestamp(start)]
        out[s] = df
        print(f"  [{i}/{len(symbols)}] {s:<12} {len(df):>6} bars  {df.index[0].date()} -> {df.index[-1].date()}  ({tag})")
    return out, failed


def synth_daily(seed=0, n=3500, start="2005-01-03", vol=0.01, scale=100.0) -> pd.DataFrame:
    """Random regime-switching prices. Tests MECHANICS only; contains no edge."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, periods=n)
    drift = np.zeros(n); i = 0
    while i < n:
        L = int(rng.integers(30, 150)); drift[i:i + L] = rng.choice([-6e-4, 0.0, 8e-4, 12e-4]); i += L
    prev, rows = scale, []
    for k in range(n):
        st = prev * math.exp(rng.normal(0, vol * .3))
        path = st * np.exp(np.cumsum(rng.normal(drift[k] / 20, vol / math.sqrt(20), 20)))
        rows.append((st, max(st, path.max()), min(st, path.min()), path[-1]))
        prev = path[-1]
    return pd.DataFrame(rows, index=idx, columns=["open", "high", "low", "close"])


def synth_universe(n=14):
    data, groups = {}, {}
    for i in range(n):
        data[f"SYN{i:02d}"] = synth_daily(i, vol=0.007 + 0.002 * (i % 4), scale=[100, 3000, 1.1, 30000][i % 4])
        groups[f"SYN{i:02d}"] = ["Synth-A", "Synth-B", "Synth-C"][i % 3]
    return data, groups


# ----------------------------------------------------------------------------- self-tests
def _mk(rows):
    d = pd.DataFrame(rows, index=pd.bdate_range("2020-01-01", periods=len(rows)), columns=["open", "high", "low", "close"])
    for k in ("ema", "macd", "above", "htf", "atr"):
        d[k] = 1.0
    return d


def unit_tests() -> bool:
    ok = True
    cfg = Cfg(slippage_bps=0.0, max_hold=15)
    filler = [(100, 101, 99, 100)] * 3
    sig = dict(t=2, side=1, kind="pullback", entry="stop", trig=100.01, pb_ext=95.0, leg=10.0, atr=2.0)

    def run(rows, name, exp_entry, exp_R, exp_reason):
        nonlocal ok
        tr = simulate(_mk(rows), [dict(sig)], cfg, "T")
        good = len(tr) == 1 and abs(tr[0]["entry"] - exp_entry) < 1e-9 and abs(tr[0]["R"] - exp_R) < 1e-9 \
            and tr[0]["reason"] == exp_reason
        got = (round(tr[0]["entry"], 3), round(tr[0]["R"], 3), tr[0]["reason"]) if tr else None
        print(f"  [{'OK' if good else 'FAIL'}] {name}: got {got} expected {(exp_entry, exp_R, exp_reason)}")
        ok &= good

    flat = [(102.5, 103, 101, 102)] * 20
    run(filler + [(102, 103, 101.5, 102.5)] + flat, "gap over trigger fills at OPEN (102), not 100.01, time stop", 102.0, 0.0, "time")
    run(filler + [(102, 103, 101.5, 102.5), (102, 110, 94, 100)] + flat, "stop AND target in one bar -> stop first", 102.0, -1.0, "stop")
    run(filler + [(102, 103, 101.5, 102.5), (102, 110, 101.9, 109), (105, 106, 101, 103)] + flat,
        "partial at 1R then breakeven stop -> +0.5R", 102.0, 0.5, "BE_stop")
    run(filler + [(99, 101, 98.5, 100.5)] + flat, "intrabar trigger fills at trigger (100.01); exit at close 102 -> (102-100.01)/5.51R", 100.01, (102 - 100.01) / (100.01 - 94.5), "time")
    return ok


def symmetry_test(raw, cfg) -> bool:
    """Long trades on the data must equal short trades on the mirrored data (price -> K - price)."""
    c0 = replace(cfg, slippage_bps=0.0)
    mir = raw.copy(); K = 2000.0
    mir["open"], mir["high"], mir["low"], mir["close"] = K - raw["open"], K - raw["low"], K - raw["high"], K - raw["close"]
    d1, d2 = prepare(raw, c0), prepare(mir, c0)
    a = simulate(d1, [s for s in generate_signals(d1, c0) if s["side"] == 1], c0)
    b = simulate(d2, [s for s in generate_signals(d2, c0) if s["side"] == -1], c0)
    ra = pd.Series({t["entry_time"]: t["R"] for t in a}); rb = pd.Series({t["entry_time"]: t["R"] for t in b})
    common = ra.index.intersection(rb.index)
    match = float(np.isclose(ra[common], rb[common], atol=1e-6).mean()) if len(common) else 0.0
    good = len(a) > 20 and abs(len(a) - len(b)) <= max(1, 0.01 * len(a)) and match > 0.99
    print(f"  [{'OK' if good else 'FAIL'}] long-vs-mirrored-short symmetry: {len(a)} vs {len(b)} trades, {match:.1%} identical R")
    return good


def causality_test(raw, cfg, cuts=(900, 1400, 1900, 2500)) -> bool:
    key = lambda s: (s["t"], s["side"], s["kind"], None if s["trig"] is None else round(float(s["trig"]), 6))
    full = {key(s) for s in generate_signals(prepare(raw, cfg), cfg)}
    ok = True
    for T in cuts:
        if T >= len(raw): continue
        part = {key(s) for s in generate_signals(prepare(raw.iloc[:T], cfg), cfg) if s["t"] <= T - 2}
        ref = {k for k in full if k[0] <= T - 2}
        same = part == ref; ok &= same
        print(f"  [{'OK' if same else 'FAIL'}] no-lookahead, data cut at bar {T}: {len(ref)} vs {len(part)} signals")
    return ok


def portfolio_scale_test(cfg) -> bool:
    raw = synth_daily(3) ; raw[["open", "high", "low", "close"]] *= 300
    d = prepare(raw, cfg); tr = pd.DataFrame(simulate(d, generate_signals(d, cfg), cfg, "X"))
    curve, skipped = portfolio(tr, cfg)
    good = len(tr) > 20 and skipped < 0.2 * len(tr)
    print(f"  [{'OK' if good else 'FAIL'}] high-priced instruments (index level ~30,000) are sized, not skipped: {len(tr) - skipped}/{len(tr)} taken")
    return good


def selftest() -> bool:
    cfg = Cfg(); raw = synth_daily(1)
    print("Unit tests of fill logic:"); r1 = unit_tests()
    print("Symmetry / causality / sizing:")
    r2 = symmetry_test(raw, cfg); r3 = causality_test(raw, cfg); r4 = portfolio_scale_test(cfg)
    allok = r1 and r2 and r3 and r4
    print("\nALL TESTS PASSED" if allok else "\n*** SOME TESTS FAILED ***")
    return allok


# =====================================================================================================================
#  PART 3 -- UNIVERSES (Yahoo symbols, grouped)
# =====================================================================================================================
UNIVERSES = {
    "US ETFs": {
        "US broad": ["SPY", "QQQ", "IWM", "DIA"],
        "US sectors": ["XLF", "XLE", "XLK", "XLV", "XLY", "XLP", "XLI", "XLU", "XLB"],
        "Intl ETFs": ["EFA", "EEM", "EWJ", "EWG", "EWU", "FXI", "EWZ", "EWY", "INDA"],
        "Bonds/Gold": ["TLT", "IEF", "GLD", "SLV"],
    },
    "World stock indices": {
        "US": ["^GSPC", "^IXIC", "^RUT", "^DJI"],
        "Europe": ["^FTSE", "^GDAXI", "^FCHI", "^STOXX50E", "^IBEX", "^AEX", "^SSMI"],
        "Asia-Pacific": ["^N225", "^HSI", "^AXJO", "^KS11", "^TWII", "^STI", "^NSEI", "^BSESN"],
        "Americas ex-US": ["^GSPTSE", "^BVSP", "^MXX"],
    },
    "International large caps": {
        "Germany": ["SAP.DE", "SIE.DE", "ALV.DE", "BAS.DE", "BMW.DE", "DTE.DE"],
        "France": ["MC.PA", "OR.PA", "AIR.PA", "SAN.PA", "TTE.PA", "BNP.PA"],
        "UK": ["HSBA.L", "SHEL.L", "AZN.L", "ULVR.L", "BP.L", "GSK.L"],
        "Switzerland/NL": ["NESN.SW", "NOVN.SW", "ROG.SW", "ASML.AS", "INGA.AS"],
        "Japan": ["7203.T", "6758.T", "9984.T", "8306.T", "6861.T"],
        "HK/China": ["0700.HK", "9988.HK", "0005.HK", "1299.HK"],
        "Korea/India/Aus": ["005930.KS", "RELIANCE.NS", "TCS.NS", "BHP.AX", "CBA.AX"],
        "Canada": ["RY.TO", "SHOP.TO", "ENB.TO"],
    },
    "US large-cap stocks": {
        "US stocks": ["AAPL", "MSFT", "AMZN", "JPM", "XOM", "JNJ", "PG", "KO", "WMT", "CAT", "GE", "INTC",
                      "CSCO", "PFE", "T", "HD", "MRK", "IBM", "BA", "DIS"],
    },
    "Commodities / FX / crypto": {
        "Futures": ["GC=F", "SI=F", "HG=F", "CL=F", "NG=F", "ZC=F", "ZW=F", "ZS=F", "KC=F", "SB=F"],
        "FX": ["EURUSD=X", "USDJPY=X", "GBPUSD=X", "AUDUSD=X", "USDCAD=X", "USDCHF=X"],
        "Crypto": ["BTC-USD", "ETH-USD"],
    },
}


def flatten(names=None):
    """-> (symbols list, {symbol: group}) for the chosen universe names (None = all)."""
    syms, groups = [], {}
    for u, gd in UNIVERSES.items():
        if names is not None and u not in names:
            continue
        for g, lst in gd.items():
            for s in lst:
                if s not in groups:
                    syms.append(s); groups[s] = g
    return syms, groups


def random_sample(n, seed, names=None):
    syms, groups = flatten(names)
    rng = random.Random(seed)
    pick = rng.sample(syms, min(n, len(syms)))
    return pick, {s: groups[s] for s in pick}


def custom(symbols):
    return list(symbols), {s: "custom" for s in symbols}


# =====================================================================================================================
#  PART 4 -- INTRADAY
# =====================================================================================================================
YF_LIMIT = {"5m": "60d", "15m": "60d", "30m": "60d", "60m": "730d"}
MIN = {"5m": 5, "15m": 15, "30m": 30, "60m": 60}


def intraday_cfg(interval="15m", flatten=None, **kw) -> Cfg:
    m = MIN[interval]
    scale = 15 / m if m < 60 else 1.0
    flat = (m < 60) if flatten is None else flatten
    base = dict(ema_len=20, atr_len=14, trend_lookback=20, trend_min_frac=0.70, use_htf=True,
                pb_window=int(16 * scale) if m < 60 else 12, pb_max_bars=int(10 * scale) if m < 60 else 8,
                leg_lookback=int(16 * scale) if m < 60 else 12, leg_min_atr=3.0,
                macd_lookback=40, ft_lookback=int(12 * scale) if m < 60 else 12,
                stop_min_atr=1.5, stop_max_atr=3.0,
                max_hold=int(16 * scale) if (m < 60 and flat) else 14,
                flatten_eod=flat, same_session=(m < 60), slippage_bps=3.0, risk_frac=0.005, max_positions=5)
    base.update(kw)
    return Cfg(**base)


# ----------------------------------------------------------------------------- data sources
def load_yf_intraday(symbols, interval="15m", cache="cache_intraday", force=False):
    """Download + MERGE into cache (so history accumulates). Returns {symbol: DataFrame (tz-aware local index)}."""
    import yfinance as yf
    os.makedirs(cache, exist_ok=True)
    out = {}
    for i, s in enumerate(symbols, 1):
        fp = os.path.join(cache, f"{_safe(s)}_{interval}.pkl")
        old = pd.read_pickle(fp) if os.path.exists(fp) else None
        fresh = old is not None and not force and (time.time() - os.path.getmtime(fp)) / 3600 < 6
        new, tag = None, ""
        if not fresh:
            for _ in range(3):
                try:
                    x = yf.download(s, period=YF_LIMIT[interval], interval=interval, auto_adjust=True,
                                    progress=False, threads=False, prepost=False)
                    if x is not None and not x.empty:
                        if isinstance(x.columns, pd.MultiIndex):
                            x.columns = x.columns.get_level_values(0)
                        x.columns = [str(c).lower() for c in x.columns]
                        new = _clean(x); break
                except Exception:
                    pass
                time.sleep(1.0)
        if new is not None and old is not None:
            df = pd.concat([old, new]); df = df[~df.index.duplicated(keep="last")].sort_index(); tag = "merged"
        elif new is not None:
            df, tag = new, "downloaded"
        elif old is not None:
            df, tag = old, "cache" if fresh else "STALE cache (download failed)"
        else:
            print(f"  [{i}/{len(symbols)}] {s:<12} FAILED"); continue
        df.to_pickle(fp)
        out[s] = df
        days = df.index.normalize().nunique()
        print(f"  [{i}/{len(symbols)}] {s:<12} {len(df):>7} bars over {days} sessions  ({tag})")
    return out


def fetch_alpaca(symbols, start, end, interval="15m", feed="iex", tz="America/New_York"):
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
    from alpaca.data.enums import Adjustment, DataFeed
    cli = StockHistoricalDataClient(os.environ["APCA_API_KEY_ID"], os.environ["APCA_API_SECRET_KEY"])
    tf = TimeFrame(MIN[interval], TimeFrameUnit.Minute) if MIN[interval] < 60 else TimeFrame(1, TimeFrameUnit.Hour)
    req = StockBarsRequest(symbol_or_symbols=list(symbols), timeframe=tf, start=pd.Timestamp(start, tz=tz),
                           end=pd.Timestamp(end, tz=tz), adjustment=Adjustment.ALL,
                           feed=DataFeed.IEX if feed == "iex" else DataFeed.SIP)
    df = cli.get_stock_bars(req).df
    out = {}
    for s in symbols:
        if s not in df.index.get_level_values(0):
            print(f"  ! no bars for {s}"); continue
        x = df.xs(s, level=0)[["open", "high", "low", "close"]]
        x.index = pd.to_datetime(x.index).tz_convert(tz)
        x = x.between_time("09:30", "15:59")                 # regular hours; Alpaca stamps bar START
        out[s] = x.sort_index()
        print(f"  {s:<8} {len(x):>8} bars")
    return out


def synth_intraday(seed=0, days=700, bars=26) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2022-01-03", periods=days)
    rows, idx, prev, drift = [], [], 100.0, 0.0
    for k, day in enumerate(dates):
        if k % 15 == 0:
            drift = rng.choice([-2e-4, 0, 3e-4, 5e-4])
        p = prev * np.exp(rng.normal(0, 0.002))
        for b in range(bars):
            st = p
            path = st * np.exp(np.cumsum(rng.normal(drift / bars, 0.0025 / np.sqrt(5), 5)))
            rows.append((st, max(st, path.max()), min(st, path.min()), path[-1]))
            idx.append(pd.Timestamp(day).tz_localize("America/New_York") + pd.Timedelta(hours=9, minutes=30 + 15 * b))
            p = path[-1]
        prev = p
    return pd.DataFrame(rows, index=pd.DatetimeIndex(idx), columns=["open", "high", "low", "close"])


# ----------------------------------------------------------------------------- preparation
def _daily_from_intraday(m):
    return m.groupby(m.index.normalize()).agg(open=("open", "first"), high=("high", "max"),
                                              low=("low", "min"), close=("close", "last"))


def prepare_intraday(m: pd.DataFrame, cfg: Cfg, daily: pd.DataFrame | None = None, skip_first=2, skip_last=3):
    d = add_indicators(m, cfg)
    d["sess"] = pd.factorize(d.index.normalize())[0]
    base = daily if daily is not None and len(daily) > 60 else _daily_from_intraday(m)
    base = base.copy(); base.index = pd.to_datetime(base.index).tz_localize(None).normalize()
    dema = base["close"].ewm(span=cfg.htf_ema, adjust=False, min_periods=cfg.htf_ema).mean()
    up = (base["close"] > dema).astype(float).where(dema.notna()).shift(1)       # yesterday's reading only
    local_days = d.index.tz_localize(None).normalize() if d.index.tz is None else \
        pd.DatetimeIndex([ts.replace(tzinfo=None) for ts in d.index.normalize()])
    d["htf"] = up.reindex(local_days, method="ffill").values
    k = d.groupby("sess").cumcount(); left = d.groupby("sess").cumcount(ascending=False)
    d["ok_entry"] = (k >= skip_first) & (left >= skip_last)
    return d


def run_intraday_study(data, groups, interval, outdir, name, daily=None, flatten=None, oos=None, slippage=3.0,
                       risk=0.005, allow_short=True, kinds=("pullback", "failtest"), baseline=False):
    cfg = intraday_cfg(interval, flatten, slippage_bps=slippage, risk_frac=risk, allow_short=allow_short)
    sf, sl = (2, 3) if MIN[interval] == 15 else ((6, 9) if MIN[interval] == 5 else (1, 1))
    prepared_input = {}
    for s, raw in data.items():
        d = prepare_intraday(raw, cfg, (daily or {}).get(s), sf, sl)
        print(f"  {s:<12} {len(d):>7} bars, median {d.groupby('sess').size().median():.0f} bars/session")
        prepared_input[s] = d
    return study(data, groups, cfg, kinds, oos, outdir, name, baseline=baseline, prepared=prepared_input)


# =====================================================================================================================
#  PART 5 -- TERMINAL UI
# =====================================================================================================================
SETTINGS_FILE = "settings.json"
DEFAULTS = dict(start="2005-01-01", oos="2019-01-01", slippage_bps=5.0, risk_frac=0.01, allow_short=True,
                use_htf=True, pullback=True, failtest=True, ft_need_keltner=False, max_positions=8,
                refresh_data=False, sample_n=30, seed=1, baseline_sims=30)
S = dict(DEFAULTS)
if os.path.exists(SETTINGS_FILE):
    try:
        S.update(json.load(open(SETTINGS_FILE)))
    except Exception:
        pass

USE_COLOR = sys.stdout.isatty() and not os.environ.get("NO_COLOR")
if os.name == "nt":
    os.system("")                       # enables ANSI colours in Windows terminals
W = 78


def _c(t, code): return f"\033[{code}m{t}\033[0m" if USE_COLOR else str(t)
def BOLD(t): return _c(t, "1")
def DIM(t): return _c(t, "2")
def RED(t): return _c(t, "31")
def GRN(t): return _c(t, "32")
def YEL(t): return _c(t, "33")
def CYN(t): return _c(t, "36")
def clear():
    if USE_COLOR:
        print("\033[2J\033[H", end="")


def save_settings():
    json.dump(S, open(SETTINGS_FILE, "w"), indent=2)


def ask(prompt, default=None):
    try:
        v = input(CYN(">> ") + prompt + (DIM(f" [{default}]") if default is not None else "") + ": ").strip()
    except (EOFError, KeyboardInterrupt):
        print(); return default
    return v if v else default


def cfg_daily() -> Cfg:
    return Cfg(slippage_bps=S["slippage_bps"], risk_frac=S["risk_frac"], allow_short=S["allow_short"],
               use_htf=S["use_htf"], ft_need_keltner=S["ft_need_keltner"], max_positions=S["max_positions"])


def kinds_on():
    k = tuple(x for x, on in (("pullback", S["pullback"]), ("failtest", S["failtest"])) if on)
    return k or ("pullback",)


def outdir_for(name):
    safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in name)
    return os.path.join("results", f"{safe}_{time.strftime('%Y%m%d_%H%M%S')}")


def _count(folder, ext):
    return len([f for f in os.listdir(folder) if f.endswith(ext)]) if os.path.isdir(folder) else 0


def banner():
    print(CYN("=" * W))
    print(CYN("  ") + BOLD("GRIMES METHOD  -  BACKTEST LAB") + DIM("   (Adam Grimes: Art & Science of Technical Analysis)"))
    print(CYN("=" * W))


def status_panel():
    onoff = lambda b: GRN("on") if b else RED("off")
    print(BOLD(" SETTINGS ") + f" slippage {S['slippage_bps']}bps/side | risk {S['risk_frac']:.2%}/trade | shorts {onoff(S['allow_short'])}"
          f" | weekly filter {onoff(S['use_htf'])}")
    print(BOLD("          ") + f" setups: {', '.join(kinds_on())} | out-of-sample from {S['oos']} | data from {S['start']}")
    print(BOLD(" DATA     ") + f" daily cache {_count('cache_daily', '.csv')} markets | intraday cache {_count('cache_intraday', '.pkl')}"
          f" | universe {len(flatten()[0])} markets | past runs {len(_runs())}")
    if LAST_RUN:
        a = LAST_RUN["avgR"]
        pas = LAST_RUN["passes"]; tot = LAST_RUN["total"]
        print(BOLD(" LAST RUN ") + f" {LAST_RUN['name']}: N={LAST_RUN['N']}, avg R " + (GRN if a > 0 else RED)(f"{a:+.3f}") +
              f", checklist " + (GRN if pas == tot else YEL if pas >= tot - 2 else RED)(f"{pas}/{tot}") + f" -> {LAST_RUN['outdir']}")
    print(DIM("-" * W))


def _runs():
    if not os.path.isdir("results"):
        return []
    r = [os.path.join("results", d) for d in os.listdir("results") if os.path.isdir(os.path.join("results", d))]
    return sorted(r, key=os.path.getmtime, reverse=True)


# ----------------------------------------------------------------------------- universe picker + loaders
UNI_NAMES = list(UNIVERSES.keys())


def pick_universe():
    print("\n" + BOLD("Choose markets") + " (comma-separated allowed, e.g. 1,2,3):")
    for i, n in enumerate(UNI_NAMES, 1):
        print(f"  {i}) {n:<28} {sum(len(v) for v in UNIVERSES[n].values())} markets")
    k = len(UNI_NAMES)
    print(f"  {k + 1}) ALL of the above ({len(flatten()[0])} markets)")
    print(f"  {k + 2}) Random sample of N markets from ALL (N={S['sample_n']}, seed={S['seed']}) - removes hand-picking")
    print(f"  {k + 3}) Type my own tickers (Yahoo symbols)")
    print(f"  {k + 4}) Synthetic demo data (tests the machinery only - NO real information)")
    c = ask("Choice", "1")
    try:
        nums = [int(x) for x in c.replace(" ", "").split(",") if x]
    except (ValueError, AttributeError):
        return None
    if k + 4 in nums:
        d, g = synth_universe(); return list(d), g, "synthetic", d
    if k + 1 in nums:
        s, g = flatten(); return s, g, "ALL", None
    if k + 2 in nums:
        s, g = random_sample(S["sample_n"], S["seed"]); return s, g, f"random{S['sample_n']}_seed{S['seed']}", None
    if k + 3 in nums:
        t = ask("Tickers separated by spaces", "SPY QQQ").replace(",", " ").split()
        s, g = custom(t); return s, g, "custom", None
    names = [UNI_NAMES[n - 1] for n in nums if 1 <= n <= k]
    if not names:
        return None
    s, g = flatten(names)
    return s, g, "+".join(n.split()[0] for n in names), None


def load_daily(sel):
    syms, groups, label, preloaded = sel
    if preloaded is not None:
        return preloaded, groups, label
    print(f"\nLoading {len(syms)} markets (cache first; re-download if older than 20h or refresh_data=on)...")
    data, failed = load_yf(syms, S["start"], force=S["refresh_data"])
    if failed:
        print(YEL(f"  could not load (skipped): {', '.join(failed)}"))
    return data, {s: groups[s] for s in data}, label


# ----------------------------------------------------------------------------- actions
def a_selftest():
    print(BOLD("\nSELF-TEST\n"))
    ok = selftest()
    print(GRN("\nEverything verified.") if ok else RED("\nSomething failed - do not trust results until fixed."))


def a_download():
    sel = pick_universe()
    if sel:
        load_daily(sel)


def a_backtest(baseline=False):
    sel = pick_universe()
    if not sel:
        return
    data, groups, label = load_daily(sel)
    if not data:
        print(RED("No data.")); return
    study(data, groups, cfg_daily(), kinds_on(), S["oos"], outdir_for(label), f"{label} daily",
          baseline=baseline, sims=S["baseline_sims"])


def a_universal():
    print("\n" + BOLD("UNIVERSALITY TEST") + ": every market in the list, pooled + per market + per region,\n"
          "out-of-sample split and a random-entry baseline. This is the main honesty check.")
    s, g = flatten()
    data, groups, label = load_daily((s, g, "ALL", None))
    if data:
        study(data, groups, cfg_daily(), kinds_on(), S["oos"], outdir_for("ALL_universal"),
              "ALL markets daily", baseline=True, sims=S["baseline_sims"])


def a_random_subsets():
    print("\n" + BOLD("RANDOM-SUBSET TEST") + ": many random subsets of markets; if the method were universal,\n"
          "almost every random subset should come out positive.")
    sel = pick_universe()
    if not sel:
        return
    data, groups, label = load_daily(sel)
    cfg = cfg_daily()
    prepared = {s: prepare(r, cfg) for s, r in data.items()}
    tr = trades_for(prepared, cfg, kinds_on())
    if tr.empty:
        print("no trades"); return
    syms = [s for s in data if (tr["symbol"] == s).any()]
    n = min(int(ask("Markets per subset", min(15, len(syms)))), len(syms))
    k = int(ask("Number of random subsets", 500))
    rng = np.random.default_rng(S["seed"])
    res = np.array([tr[tr["symbol"].isin(rng.choice(syms, n, replace=False))]["R"].mean() for _ in range(k)])
    pos = np.mean(res > 0)
    print(f"\nMean R over {k} random subsets of {n} markets: median {np.median(res):+.3f}, "
          f"5th pct {np.percentile(res, 5):+.3f}, 95th pct {np.percentile(res, 95):+.3f}; "
          + (GRN if pos > .9 else YEL if pos > .6 else RED)(f"positive in {pos:.0%} of subsets"))
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        od = outdir_for(label + "_subsets"); os.makedirs(od, exist_ok=True)
        plt.figure(figsize=(9, 5)); plt.hist(res, bins=40, color="steelblue"); plt.axvline(0, color="k")
        plt.title(f"avg R across {k} random subsets of {n} markets ({label})"); plt.grid(alpha=.3)
        plt.savefig(os.path.join(od, "subsets.png"), dpi=110); plt.close()
        pd.Series(res, name="avgR").to_csv(os.path.join(od, "subsets.csv"), index=False)
        print(f"saved {od}/subsets.png and subsets.csv")
    except Exception as ex:
        print("plot failed:", ex)


def a_grid():
    sel = pick_universe()
    if not sel:
        return
    data, groups, label = load_daily(sel)
    cfg = cfg_daily()
    prepared = {s: prepare(r, cfg) for s, r in data.items()}
    od = outdir_for(label + "_grid"); os.makedirs(od, exist_ok=True)
    for (n1, v1, n2, v2) in (("leg_min_atr", [3.0, 4.0, 5.0, 6.0], "stop_min_atr", [1.5, 2.0, 2.5, 3.0]),
                             ("trend_min_frac", [0.6, 0.7, 0.8], "max_hold", [8, 15, 25])):
        avg, cnt = param_grid(prepared, cfg, kinds_on(), n1, v1, n2, v2)
        print(f"\n{BOLD('Average R')}: {n1} (rows) x {n2} (columns)\n{avg.to_string()}\nNumber of trades:\n{cnt.to_string()}")
        avg.to_csv(os.path.join(od, f"grid_{n1}_{n2}.csv"))
    print("\nA real edge is positive across NEIGHBOURING cells. One lonely positive cell = curve fit.")
    print(f"saved to {od}/")


def a_intraday_yf():
    sel = pick_universe()
    if not sel:
        return
    syms, groups, label, pre = sel
    if pre is not None:
        print(YEL("Synthetic demo is daily-only; pick real markets for intraday.")); return
    iv = ask("Bar size  5m / 15m / 60m", "15m")
    if iv not in MIN:
        print(RED("bad interval")); return
    fl = ask("Close everything at end of session? y/n", "y" if iv != "60m" else "n").lower().startswith("y")
    print(f"\nyfinance only provides the last {YF_LIMIT[iv]} of {iv} bars. Each run MERGES into cache_intraday/, so the\n"
          "history grows if you re-run weekly. For years of history use Alpaca (menu 9).")
    data = load_yf_intraday(syms, iv, force=S["refresh_data"])
    if not data:
        print(RED("No data.")); return
    print("Loading daily history for the higher-time-frame filter...")
    daily, _ = load_yf(list(data), "2010-01-01")
    g = {s: groups.get(s, "") for s in data}
    run_intraday_study(data, g, iv, outdir_for(f"{label}_{iv}"), f"{label} intraday {iv} (yfinance)", daily, fl, None,
                       S["slippage_bps"] if S["slippage_bps"] != 5.0 else 3.0, 0.005, S["allow_short"], kinds_on())


def a_intraday_alpaca():
    if not (os.environ.get("APCA_API_KEY_ID") and os.environ.get("APCA_API_SECRET_KEY")):
        print(YEL("\nSet your keys first (in the terminal, never in chat):\n"
                  "  export APCA_API_KEY_ID=...   export APCA_API_SECRET_KEY=...   (Windows: setx / set)\n"
                  "  pip install alpaca-py\nthen choose this option again.")); return
    syms = ask("US symbols", "SPY QQQ IWM DIA XLF XLE XLK").replace(",", " ").split()
    start = ask("Start date", "2023-01-02"); end = ask("End date", time.strftime("%Y-%m-%d"))
    iv = ask("Bar size  5m / 15m / 60m", "15m"); feed = ask("Feed iex / sip", "iex")
    data = fetch_alpaca(syms, start, end, iv, feed)
    if not data:
        print(RED("No data.")); return
    run_intraday_study(data, {s: "alpaca" for s in data}, iv, outdir_for(f"alpaca_{iv}"), f"Alpaca {feed} {iv}",
                       None, None, None, 3.0, 0.005, S["allow_short"], kinds_on())


def a_settings():
    keys = list(DEFAULTS)
    while True:
        print("\n" + BOLD("SETTINGS"))
        for i, k in enumerate(keys, 1):
            print(f"  {i:>2}) {k:<16} = {YEL(S[k])}")
        print("   r) reset to defaults     0) back")
        c = ask("Edit which", "0")
        if c == "0":
            save_settings(); return
        if c and c.lower() == "r":
            S.update(DEFAULTS); continue
        try:
            k = keys[int(c) - 1]
        except Exception:
            continue
        cur = S[k]
        v = ask(f"New value for {k}", cur)
        try:
            if isinstance(cur, bool):
                nv = str(v).lower() in ("1", "y", "yes", "true", "on")
            elif isinstance(cur, int):
                nv = int(v)
            elif isinstance(cur, float):
                nv = float(v)
            else:
                nv = str(v)
            if k in ("start", "oos"):
                nv = pd.Timestamp(nv).strftime("%Y-%m-%d")                 # must be a real date
            ranges = dict(risk_frac=(0.0005, 0.05), slippage_bps=(0, 100), max_positions=(1, 50),
                          sample_n=(3, 500), baseline_sims=(5, 500))
            if k in ranges and not (ranges[k][0] <= nv <= ranges[k][1]):
                raise ValueError(f"{k} must be between {ranges[k][0]} and {ranges[k][1]}")
            S[k] = nv
        except Exception as ex:
            print(RED(f"rejected: {ex}"))


def a_view_report():
    runs = _runs()
    if not runs:
        print("No past runs yet."); return
    print("\n" + BOLD("Past runs") + " (newest first):")
    for i, r in enumerate(runs[:20], 1):
        print(f"  {i:>2}) {os.path.basename(r)}")
    c = ask("Open which (number, Enter = cancel)", None)
    try:
        r = runs[int(c) - 1]
    except Exception:
        return
    fp = os.path.join(r, "report.txt")
    print(DIM("-" * W))
    print(open(fp).read() if os.path.exists(fp) else "(no report.txt in this folder)")
    print(DIM("-" * W) + f"\nfolder: {os.path.abspath(r)}  (equity.png, scoreboard.png, trades.csv inside)")


def a_help():
    print(f"""
{BOLD('HOW TO READ THE RESULTS')}
  R        profit divided by the initial risk of the trade. -1R = stopped out, +2R = won twice what you risked.
  avg R    expectancy per trade. Above 0 AFTER costs is the minimum for an edge. 0.1R is small but real.
  t-stat   avg R divided by its standard error. >2 is 'maybe', >=3 is 'strong' (we test many things, so be strict).
  win %    meaningless alone - a 35% win rate can be great with 3R winners, a 70% rate can lose money.
  PF       profit factor = gross wins / gross losses.
  OOS      out-of-sample: trades after the split date. If in-sample is great and OOS is flat, it was fitted/luck.
  baseline random entries with the SAME exits; if the strategy does not beat them the entry adds nothing.

{BOLD('THE SETUPS')}
  pullback   in a trending market (20-bar majority above/below the 20 EMA, weekly trend agrees) after a strong
             leg, buy the first quiet 25-75% retracement when price breaks the signal bar's high.
  failtest   price probes below a 20-bar low and closes back above it (spring); mirrored for highs.
  exits      stop >= 2 ATR, half off at +1R, rest to measured move / 3R with breakeven stop, 15-bar time stop.

{BOLD('HONESTY RULES BUILT IN')}
  signals use only closed bars; fills happen on LATER bars; gaps fill at the open; stop beats target inside a bar.
  Menu 1 proves it. Results on synthetic data mean nothing about markets - they only test the machinery.
  Past performance, even honestly measured, is not a promise. Paper-trade before real money.
""")


MENU = [
    ("TESTS", None, None),
    ("1", "Self-test (no-lookahead, fills, symmetry, sizing)", a_selftest),
    ("DATA", None, None),
    ("2", "Download / refresh daily data for a market set", a_download),
    ("DAILY BACKTESTS", None, None),
    ("3", "Backtest a market set", lambda: a_backtest(False)),
    ("4", "Backtest + random-entry baseline", lambda: a_backtest(True)),
    ("5", "UNIVERSALITY TEST - all markets world-wide (main honesty check)", a_universal),
    ("IS IT REAL OR LUCK?", None, None),
    ("6", "Random-subset test (hand-picked luck?)", a_random_subsets),
    ("7", "Parameter robustness grid (curve-fit?)", a_grid),
    ("INTRADAY", None, None),
    ("8", "Intraday with yfinance (5m / 15m / 60m)", a_intraday_yf),
    ("9", "Intraday with Alpaca (needs API keys)", a_intraday_alpaca),
    ("TOOLS", None, None),
    ("10", "Settings (costs, risk, shorts, dates, setups)", a_settings),
    ("11", "View a previous report", a_view_report),
    ("12", "Help: what do these numbers mean?", a_help),
    ("0", "Exit", None),
]


def main():
    actions = {k: f for k, _, f in MENU if f}
    while True:
        clear(); banner(); status_panel()
        for k, t, f in MENU:
            if f is None and t is None:
                print("\n " + YEL(k))
            else:
                print(f"   {BOLD(k.rjust(2))}  {t}")
        c = ask("\nChoose a number", "0")
        if c in ("0", "q", "exit", None):
            print("bye"); return
        fn = actions.get(c)
        if not fn:
            continue
        try:
            fn()
        except KeyboardInterrupt:
            print(YEL("\ninterrupted"))
        except Exception as ex:
            import traceback; traceback.print_exc()
            print(RED(f"\n!! {type(ex).__name__}: {ex}"))
        ask("\nPress Enter to return to the menu")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(0 if selftest() else 1)
    if "--demo" in sys.argv:        # non-interactive smoke test on synthetic data
        d, g = synth_universe()
        study(d, g, Cfg(), ("pullback", "failtest"), "2016-01-01", outdir_for("demo"), "demo synthetic", baseline=True, sims=10)
        sys.exit(0)
    main()
