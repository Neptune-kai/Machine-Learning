import os
import json
from datetime import datetime, timedelta, timezone
import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv

from sklearn.linear_model import Ridge, RidgeCV
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import make_pipeline

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, GetOrdersRequest
from alpaca.trading.enums import OrderSide, TimeInForce, QueryOrderStatus, AccountStatus
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.data.enums import DataFeed

# 1. Load Credentials
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_PATH = os.path.join(SCRIPT_DIR, ".env")
load_dotenv(dotenv_path=ENV_PATH, override=True)

API_KEY = os.getenv("ALPACA_API_KEY")
SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
BASE_URL = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")

if not API_KEY or not SECRET_KEY:
    raise ValueError("Missing ALPACA_API_KEY or ALPACA_SECRET_KEY in .env or GitHub Secrets.")

IS_PAPER = "paper" in BASE_URL.lower()
trading_client = TradingClient(API_KEY, SECRET_KEY, paper=IS_PAPER)
data_client = StockHistoricalDataClient(API_KEY, SECRET_KEY)

# 2. File & Universe Configuration
META_MODEL_PATH = os.path.join(SCRIPT_DIR, "meta_ensemble.joblib")
ALGO_STATS_PATH = os.path.join(SCRIPT_DIR, "algo_leaderboard.json")

FEATURES = [
    "return_1", "return_3", "return_6", "return_12",
    "hl_spread", "oc_change", "vol_ratio", "vwap_dist",
    "volatility_12", "rsi_14", "ema_diff", "macd_hist", "bb_zscore"
]

WATCHLIST = [
    "TSLA", "NVDA", "AAPL", "AMD", "PLTR", "AMZN", "META", "MSFT", "GOOGL", "GOOG",
    "SOFI", "HOOD", "RKLB", "SMCI", "MSTR", "INTC", "MU", "AVGO", "NFLX", "UBER",
    "PYPL", "HIMS", "IONQ", "RIOT", "MARA", "CLSK", "ACHR", "JOBY", "COIN", "SQ",
    "SHOP", "SNOW", "CRWD", "PANW", "NET", "DDOG", "ZS", "CRM", "ORCL", "QCOM",
    "ARM", "TSM", "MRVL", "DELL", "BABA", "NIO", "RIVN", "LCID", "F", "GM",
    "DIS", "BA", "AAL", "DAL", "CCL", "NCLH", "JPM", "BAC", "WFC", "C"
]


# ==========================================
# PART A: TECHNICAL INDICATORS & DATA
# ==========================================
def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("timestamp").copy()
    close = df["close"]

    df["return_1"] = close.pct_change(1)
    df["return_3"] = close.pct_change(3)
    df["return_6"] = close.pct_change(6)
    df["return_12"] = close.pct_change(12)
    df["hl_spread"] = (df["high"] - df["low"]) / close
    df["oc_change"] = (close - df["open"]) / df["open"]
    df["vol_ratio"] = df["volume"] / (df["volume"].rolling(12).mean() + 1e-6)
    df["volatility_12"] = df["return_1"].rolling(12).std()

    # Intraday VWAP distance
    typical_price = (df["high"] + df["low"] + close) / 3.0
    cum_vp = (typical_price * df["volume"]).rolling(24, min_periods=1).sum()
    cum_vol = df["volume"].rolling(24, min_periods=1).sum() + 1e-6
    vwap = cum_vp / cum_vol
    df["vwap_dist"] = (close - vwap) / vwap

    # RSI-14
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean() + 1e-6
    rs = gain / loss
    df["rsi_14"] = 100.0 - (100.0 / (1.0 + rs))

    # EMA 9 vs EMA 21
    ema_9 = close.ewm(span=9, adjust=False).mean()
    ema_21 = close.ewm(span=21, adjust=False).mean()
    df["ema_diff"] = (ema_9 - ema_21) / ema_21

    # MACD Histogram
    ema_12 = close.ewm(span=12, adjust=False).mean()
    ema_26 = close.ewm(span=26, adjust=False).mean()
    macd = ema_12 - ema_26
    signal_line = macd.ewm(span=9, adjust=False).mean()
    df["macd_hist"] = (macd - signal_line) / close

    # Bollinger Band Z-Score (20-bar)
    sma_20 = close.rolling(20).mean()
    std_20 = close.rolling(20).std() + 1e-6
    df["bb_zscore"] = (close - sma_20) / std_20

    # Rolling 24-bar high pullback
    rolling_high = df["high"].rolling(24, min_periods=1).max()
    df["drop_from_high"] = (close - rolling_high) / rolling_high
    return df


def fetch_historical_bars(symbols: list[str], days_back: int = 21) -> pd.DataFrame:
    end_dt = datetime.now(timezone.utc)
    start_dt = end_dt - timedelta(days=days_back)

    req = StockBarsRequest(
        symbol_or_symbols=symbols,
        timeframe=TimeFrame(5, TimeFrameUnit.Minute),
        start=start_dt,
        end=end_dt,
        feed=DataFeed.IEX,
    )
    bars_df = data_client.get_stock_bars(req).df.reset_index()
    bars_df["timestamp"] = pd.to_datetime(bars_df["timestamp"], utc=True)

    times = bars_df["timestamp"].dt.time
    rth_start = datetime.strptime("14:30:00", "%H:%M:%S").time()
    rth_end = datetime.strptime("21:00:00", "%H:%M:%S").time()
    bars_df = bars_df[(times >= rth_start) & (times < rth_end)].copy()
    bars_df["trade_date"] = bars_df["timestamp"].dt.date
    return bars_df


# ==========================================
# PART B: 20 DISTINCT TRADING ALGORITHMS
# ==========================================
def generate_20_algo_signals(df: pd.DataFrame, ml_models: dict) -> pd.DataFrame:
    """
    Computes continuous directional signals (-1.0 to +1.0) for all 20 trading algorithms.
    Positive = Bullish / Buy signal; Negative = Bearish / Sell signal.
    """
    s = pd.DataFrame(index=df.index)

    # 1. Fast Momentum
    s["algo_01_fast_momentum"] = np.tanh((df["return_1"] + df["return_3"]) * 150)

    # 2. Trend Continuation
    s["algo_02_trend_continuation"] = np.tanh((df["return_6"] + df["return_12"]) * 80)

    # 3. VWAP Mean Reversion
    s["algo_03_vwap_reversion"] = np.tanh(-df["vwap_dist"] * 200)

    # 4. VWAP Volume Breakout
    s["algo_04_vwap_breakout"] = np.where(
        (df["vwap_dist"] > 0) & (df["vol_ratio"] > 1.3),
        np.tanh(df["vwap_dist"] * df["vol_ratio"] * 150),
        -0.2
    )

    # 5. Bollinger Squeeze Expansion
    s["algo_05_bollinger_squeeze"] = np.where(
        df["volatility_12"] < df["volatility_12"].median(),
        np.tanh(df["return_1"] * 250),
        0.0
    )

    # 6. Upper Bollinger Breakout
    s["algo_06_bb_breakout"] = np.tanh((df["bb_zscore"] - 1.0) * 0.8)

    # 7. Oversold RSI Bounce
    s["algo_07_rsi_oversold"] = np.tanh((35.0 - df["rsi_14"]) / 15.0)

    # 8. RSI Momentum Thrust
    s["algo_08_rsi_momentum"] = np.tanh((df["rsi_14"] - 55.0) / 15.0)

    # 9. EMA 9/21 Crossover
    s["algo_09_ema_cross"] = np.tanh(df["ema_diff"] * 300)

    # 10. EMA Ribbon Trend + Volume
    s["algo_10_ema_ribbon"] = np.tanh(df["ema_diff"] * 200 + (df["vol_ratio"] - 1.0) * 0.3)

    # 11. MACD Histogram Surge
    s["algo_11_macd_surge"] = np.tanh(df["macd_hist"] * 500)

    # 12. Volume Climax Breakout
    s["algo_12_volume_climax"] = np.tanh((df["vol_ratio"] - 1.5) * np.sign(df["oc_change"]) * 0.8)

    # 13. Rolling High Breakout
    s["algo_13_high_breakout"] = np.where(df["drop_from_high"] >= -0.001, 0.8, -0.2)

    # 14. Shallow Pullback in Uptrend
    s["algo_14_shallow_pullback"] = np.where(
        (df["ema_diff"] > 0) & (df["drop_from_high"].between(-0.008, -0.001)),
        0.85,
        -0.1
    )

    # 15. Volatility Expansion Directional
    s["algo_15_vol_expansion"] = np.tanh(df["hl_spread"] * np.sign(df["oc_change"]) * 120)

    # 16. Mean-Reverting Sharp Drop Fade
    s["algo_16_gap_fade"] = np.where(df["return_3"] < -0.008, 0.9, np.tanh(-df["return_3"] * 50))

    # 17–20. Machine Learning Sub-Models (scaled to [-1, 1])
    X = df[FEATURES]
    s["algo_17_ridge_ml"] = np.tanh(ml_models["ridge"].predict(X) * 400)
    s["algo_18_random_forest_ml"] = np.tanh(ml_models["rf"].predict(X) * 400)
    s["algo_19_grad_boost_ml"] = np.tanh(ml_models["gbr"].predict(X) * 400)
    s["algo_20_neural_net_ml"] = np.tanh(ml_models["mlp"].predict(X) * 400)

    return s


# ==========================================
# PART C: META-LEARNER TRAINING & EVALUATION
# ==========================================
def train_20_algo_meta_learner(raw_df: pd.DataFrame):
    chunks = []
    for (_, _), group in raw_df.groupby(["symbol", "trade_date"]):
        if len(group) < 25:
            continue
        feat_df = compute_features(group)
        feat_df["target_return"] = feat_df["close"].pct_change(3).shift(-3)
        chunks.append(feat_df.dropna())

    full_df = pd.concat(chunks, ignore_index=True).sort_values("timestamp")
    split_idx = int(len(full_df) * 0.65)
    base_train_df = full_df.iloc[:split_idx]
    meta_train_df = full_df.iloc[split_idx:]

    print(f"Training 4 base ML algorithms on {len(base_train_df):,} bars...")
    ml_models = {
        "ridge": make_pipeline(RobustScaler(), Ridge(alpha=10.0)),
        "rf": RandomForestRegressor(n_estimators=60, max_depth=6, random_state=42, n_jobs=-1),
        "gbr": HistGradientBoostingRegressor(max_iter=80, max_depth=4, random_state=42),
        "mlp": make_pipeline(RobustScaler(), MLPRegressor(hidden_layer_sizes=(32, 16), max_iter=150, random_state=42)),
    }

    for name, mdl in ml_models.items():
        mdl.fit(base_train_df[FEATURES], base_train_df["target_return"])

    print("Evaluating all 20 trading algorithms on out-of-sample market data...")
    algo_signals = generate_20_algo_signals(meta_train_df, ml_models)
    y_future = meta_train_df["target_return"].values

    # Evaluate individual Sharpe-like return of each of the 20 algorithms
    leaderboard = {}
    for col in algo_signals.columns:
        sig = algo_signals[col].values
        # Simulated return when algorithm votes bullish (> 0.25)
        active_returns = y_future[sig > 0.25]
        if len(active_returns) >= 15:
            sharpe = float(np.mean(active_returns) / (np.std(active_returns) + 1e-6))
            win_rate = float(np.mean(active_returns > 0))
        else:
            sharpe = -1.0
            win_rate = 0.0
        leaderboard[col] = {
            "sharpe": round(sharpe, 4),
            "win_rate": round(win_rate, 4),
            "signals_fired": int(np.sum(sig > 0.25)),
        }

    # Meta-Learner: RidgeCV learns optimal combination weights across all 20 algorithms
    meta_learner = RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0])
    meta_learner.fit(algo_signals, y_future)

    # Save learned weights and leaderboard
    coef_map = dict(zip(algo_signals.columns, [round(float(c), 6) for c in meta_learner.coef_]))
    for algo_name in leaderboard:
        leaderboard[algo_name]["meta_weight"] = coef_map[algo_name]

    ranked = sorted(leaderboard.items(), key=lambda x: x[1]["sharpe"], reverse=True)
    print("\n--- TOP 5 PERFORMING ALGORITHMS (OUT-OF-SAMPLE) ---")
    for name, stats in ranked[:5]:
        print(f"  {name:28s} | Sharpe: {stats['sharpe']:+.3f} | WinRate: {stats['win_rate']*100:.1f}% | Weight: {stats['meta_weight']:+.5f}")

    with open(ALGO_STATS_PATH, "w") as f:
        json.dump(dict(ranked), f, indent=2)

    bundle = {
        "ml_models": ml_models,
        "meta_learner": meta_learner,
        "algo_columns": list(algo_signals.columns),
        "pred_std": float(np.std(meta_learner.predict(algo_signals))) + 1e-6,
    }
    joblib.dump(bundle, META_MODEL_PATH)
    return bundle


# ==========================================
# PART D: LIVE / PAPER ENSEMBLE EXECUTION
# ==========================================
def get_live_ensemble_predictions(symbols: list[str], bundle: dict) -> dict[str, dict]:
    if not symbols:
        return {}

    now = datetime.now(timezone.utc)
    req = StockBarsRequest(
        symbol_or_symbols=symbols,
        timeframe=TimeFrame(5, TimeFrameUnit.Minute),
        start=now - timedelta(hours=18),
        end=now,
        feed=DataFeed.IEX,
    )
    bars_df = data_client.get_stock_bars(req).df
    if bars_df.empty:
        return {}

    bars_df = bars_df.reset_index()
    latest_rows = []
    meta_info = []

    for symbol, group in bars_df.groupby("symbol"):
        if len(group) < 25:
            continue
        feat_df = compute_features(group)
        latest = feat_df.iloc[-1]
        if latest[FEATURES].isna().any():
            continue
        latest_rows.append(feat_df.iloc[[-1]])
        meta_info.append((symbol, float(latest["close"])))

    if not latest_rows:
        return {}

    batch_df = pd.concat(latest_rows, ignore_index=True)
    algo_matrix = generate_20_algo_signals(batch_df, bundle["ml_models"])
    meta_preds = bundle["meta_learner"].predict(algo_matrix[bundle["algo_columns"]])

    results = {}
    for idx, (symbol, price) in enumerate(meta_info):
        algo_votes = algo_matrix.iloc[idx]
        bullish_count = int((algo_votes > 0.25).sum())
        bearish_count = int((algo_votes < -0.25).sum())
        results[symbol] = {
            "meta_pred": float(meta_preds[idx]),
            "price": price,
            "bullish_algos": bullish_count,
            "bearish_algos": bearish_count,
        }
    return results


def run_ensemble_bot(bundle: dict):
    account = trading_client.get_account()
    equity = float(account.equity)
    buying_power = float(account.buying_power)
    print(f"\nAccount #{account.account_number} | Status: {account.status.value} | Equity: ${equity:,.2f} | Buying Power: ${buying_power:,.2f}")

    if account.status != AccountStatus.ACTIVE or account.trading_blocked:
        raise RuntimeError(f"Account {account.account_number} cannot trade (status={account.status.value}).")

    clock = trading_client.get_clock()
    if not clock.is_open:
        print("Market is currently closed. 20-algorithm training & leaderboard update complete.")
        return

    held_positions = trading_client.get_all_positions()
    held_symbols = {p.symbol for p in held_positions}
    all_symbols = list(set(WATCHLIST) | held_symbols)

    predictions = get_live_ensemble_predictions(all_symbols, bundle)
    buy_threshold = bundle["pred_std"] * 0.5
    sell_threshold = -bundle["pred_std"] * 0.5
    print(f"Scored {len(predictions)} symbols across 20 algorithms | Meta Buy Threshold: {buy_threshold:.6f}")

    top_5 = sorted(predictions.items(), key=lambda x: x[1]["meta_pred"], reverse=True)[:5]
    for sym, info in top_5:
        print(
            f"  Top Ensemble -> {sym}: meta_pred={info['meta_pred']:+.6f} | "
            f"Votes: {info['bullish_algos']} Bull / {info['bearish_algos']} Bear @ ${info['price']:.2f}"
        )

    # 1. Exit positions when Meta-Learner flips bearish or stop-loss/take-profit triggers
    for p in held_positions:
        pl_pct = float(p.unrealized_plpc)
        info = predictions.get(p.symbol)

        if pl_pct <= -0.015:
            print(f"ENSEMBLE STOP-LOSS {p.symbol}: {pl_pct * 100:.2f}%")
            trading_client.close_position(p.symbol)
        elif pl_pct >= 0.025:
            print(f"ENSEMBLE TAKE-PROFIT {p.symbol}: +{pl_pct * 100:.2f}%")
            trading_client.close_position(p.symbol)
        elif info and info["meta_pred"] <= sell_threshold:
            print(f"ENSEMBLE EXIT {p.symbol}: meta_pred={info['meta_pred']:+.6f} ({info['bearish_algos']}/20 bearish)")
            trading_client.close_position(p.symbol)

    # 2. Enter top consensus signals
    open_orders = trading_client.get_orders(GetOrdersRequest(status=QueryOrderStatus.OPEN))
    excluded = held_symbols | {o.symbol for o in open_orders}

    candidates = sorted(
        [
            (sym, info)
            for sym, info in predictions.items()
            if sym not in excluded and info["meta_pred"] >= buy_threshold and info["price"] > 1.0
        ],
        key=lambda x: x[1]["meta_pred"],
        reverse=True,
    )

    position_budget = equity * 0.05  # 5% of equity per position
    orders_placed = 0

    for symbol, info in candidates[:10]:
        if buying_power < position_budget or buying_power < info["price"]:
            break

        qty = int(position_budget // info["price"])
        if qty > 0:
            print(
                f"ENSEMBLE BUY: {symbol} @ ${info['price']:.2f} | "
                f"Meta-Pred: {info['meta_pred']:+.6f} ({info['bullish_algos']}/20 bullish) | Qty: {qty}"
            )
            trading_client.submit_order(
                MarketOrderRequest(
                    symbol=symbol,
                    qty=qty,
                    side=OrderSide.BUY,
                    time_in_force=TimeInForce.DAY,
                )
            )
            buying_power -= qty * info["price"]
            orders_placed += 1

    print(f"Run complete. Ensemble orders placed: {orders_placed}")


if __name__ == "__main__":
    print("Fetching 21 days of 5-minute bars to train 20-Algorithm Meta-Learner...")
    raw_history = fetch_historical_bars(WATCHLIST[:30], days_back=21)
    ensemble_bundle = train_20_algo_meta_learner(raw_history)

    print(f"Connected to Alpaca ({'PAPER' if IS_PAPER else 'LIVE'} mode).")
    run_ensemble_bot(ensemble_bundle)
    print(f"Connected to Alpaca ({'PAPER' if IS_PAPER else 'LIVE'} mode).")
    run_autonomous_bot(ridge_model, autonomous_params)
