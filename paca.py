import os
import json
import random
from datetime import datetime, timedelta, timezone
import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.linear_model import Ridge
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
MODEL_PATH = os.path.join(SCRIPT_DIR, "ridge_model.joblib")
PARAMS_PATH = os.path.join(SCRIPT_DIR, "best_params.json")

FEATURES = [
    "return_1", "return_3", "return_6", "return_12",
    "hl_spread", "oc_change", "vol_ratio", "vwap_dist", "volatility_12"
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
# PART A: SELF-LEARNING FEATURE & MODEL ENGINE
# ==========================================
def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("timestamp").copy()
    df["return_1"] = df["close"].pct_change(1)
    df["return_3"] = df["close"].pct_change(3)
    df["return_6"] = df["close"].pct_change(6)
    df["return_12"] = df["close"].pct_change(12)
    df["hl_spread"] = (df["high"] - df["low"]) / df["close"]
    df["oc_change"] = (df["close"] - df["open"]) / df["open"]
    df["vol_ratio"] = df["volume"] / (df["volume"].rolling(12).mean() + 1e-6)
    df["volatility_12"] = df["return_1"].rolling(12).std()

    # Intraday VWAP approximation distance
    typical_price = (df["high"] + df["low"] + df["close"]) / 3.0
    cum_vp = (typical_price * df["volume"]).rolling(24, min_periods=1).sum()
    cum_vol = df["volume"].rolling(24, min_periods=1).sum() + 1e-6
    vwap = cum_vp / cum_vol
    df["vwap_dist"] = (df["close"] - vwap) / vwap
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


def train_fresh_model(raw_df: pd.DataFrame):
    """Retrains the model on the latest market data every run so it continuously adapts."""
    chunks = []
    for (_, _), group in raw_df.groupby(["symbol", "trade_date"]):
        if len(group) < 20:
            continue
        feat_df = compute_features(group)
        # Predict 3-bar (15-min) forward return to capture stronger moves over noise
        feat_df["target_return"] = feat_df["close"].pct_change(3).shift(-3)
        chunks.append(feat_df.dropna())

    full_df = pd.concat(chunks, ignore_index=True).sort_values("timestamp")

    # Model selects its own optimal regularization strength via validation split
    split_idx = int(len(full_df) * 0.8)
    train_df, val_df = full_df.iloc[:split_idx], full_df.iloc[split_idx:]

    best_alpha = 1.0
    best_mse = float("inf")
    for alpha in [0.01, 0.1, 1.0, 10.0, 100.0]:
        candidate = make_pipeline(RobustScaler(), Ridge(alpha=alpha))
        candidate.fit(train_df[FEATURES], train_df["target_return"])
        preds = candidate.predict(val_df[FEATURES])
        mse = float(np.mean((preds - val_df["target_return"]) ** 2))
        if mse < best_mse:
            best_mse = mse
            best_alpha = alpha

    final_model = make_pipeline(RobustScaler(), Ridge(alpha=best_alpha))
    final_model.fit(full_df[FEATURES], full_df["target_return"])
    joblib.dump(final_model, MODEL_PATH)
    print(f"Self-trained fresh model on {len(full_df):,} bars (auto-selected alpha={best_alpha}).")
    return final_model, full_df


# ==========================================
# PART B: UNCONSTRAINED GENETIC ALGORITHM
# ==========================================
def evaluate_fitness(df: pd.DataFrame, genes: dict) -> float:
    buy_thresh = genes["buy_thresh"]
    sell_thresh = genes["sell_thresh"]
    stop_loss = genes["stop_loss"]
    take_profit = genes["take_profit"]
    max_hold = int(round(genes["max_hold_bars"]))
    vol_boost = genes["vol_weight"]

    trade_returns = []

    for (_, _), day_df in df.groupby(["symbol", "trade_date"]):
        in_position = False
        entry_price = 0.0
        bars_held = 0

        closes = day_df["close"].values
        preds = day_df["pred_return"].values
        vol_ratios = day_df["vol_ratio"].values

        for i in range(len(day_df)):
            price = closes[i]
            # Model combines raw return prediction with evolved volume sensitivity
            signal = preds[i] + vol_boost * (vol_ratios[i] - 1.0) * 0.001
            is_last_bar = (i == len(day_df) - 1)

            if not in_position and not is_last_bar:
                if signal >= buy_thresh:
                    in_position = True
                    entry_price = price
                    bars_held = 0
            elif in_position:
                bars_held += 1
                current_pl = (price - entry_price) / entry_price
                if (
                    current_pl <= stop_loss
                    or current_pl >= take_profit
                    or signal <= sell_thresh
                    or bars_held >= max_hold
                    or is_last_bar
                ):
                    trade_returns.append(current_pl - 0.0002)
                    in_position = False

    if len(trade_returns) < 10:
        return -999.0

    returns_arr = np.array(trade_returns)
    # Pure risk-adjusted total return (no artificial trade count caps)
    sharpe = np.mean(returns_arr) / (np.std(returns_arr) + 1e-6)
    return float(sharpe * np.log1p(len(returns_arr)))


def evolve_unconstrained_strategy(clean_df: pd.DataFrame, pop_size: int = 30, generations: int = 40) -> dict:
    print("Evolving unconstrained strategy parameters from scratch...")
    clean_df = clean_df.copy()

    n = len(clean_df)
    train_df = clean_df.iloc[:int(n * 0.7)]
    val_df = clean_df.iloc[int(n * 0.7):]

    def random_individual() -> dict:
        return {
            "buy_thresh": random.uniform(-0.001, 0.005),
            "sell_thresh": random.uniform(-0.005, 0.001),
            "stop_loss": random.uniform(-0.05, -0.002),
            "take_profit": random.uniform(0.002, 0.08),
            "max_hold_bars": random.uniform(2, 78),
            "allocation_pct": random.uniform(0.02, 0.25),
            "vol_weight": random.uniform(-1.0, 1.0),
        }

    # Seed population with previous evolved genes if compatible (continuous evolution)
    population = [random_individual() for _ in range(pop_size)]
    if os.path.exists(PARAMS_PATH):
        try:
            with open(PARAMS_PATH, "r") as f:
                prior = json.load(f)
            if set(population[0].keys()).issubset(prior.keys()):
                population[0] = {k: prior[k] for k in population[0]}
        except Exception:
            pass

    best_val_score = -999.0
    best_genes = population[0]

    for _ in range(generations):
        scored = sorted([(evaluate_fitness(train_df, ind), ind) for ind in population], key=lambda x: x[0], reverse=True)
        survivors = [ind for _, ind in scored[:6]]

        val_score = evaluate_fitness(val_df, survivors[0])
        if val_score > best_val_score:
            best_val_score = val_score
            best_genes = survivors[0].copy()

        next_gen = survivors.copy()
        while len(next_gen) < pop_size:
            p1, p2 = random.sample(survivors, 2)
            # Blend crossover + unconstrained mutation
            alpha = random.random()
            child = {k: alpha * p1[k] + (1 - alpha) * p2[k] for k in p1}
            for k in child:
                if random.random() < 0.30:
                    child[k] *= random.uniform(0.75, 1.25)
                    if k in ("buy_thresh", "sell_thresh", "vol_weight") and random.random() < 0.2:
                        child[k] += random.uniform(-0.0005, 0.0005)
            child["stop_loss"] = min(child["stop_loss"], -0.001)
            child["take_profit"] = max(child["take_profit"], 0.001)
            child["max_hold_bars"] = float(np.clip(child["max_hold_bars"], 1, 78))
            child["allocation_pct"] = float(np.clip(child["allocation_pct"], 0.01, 0.50))
            next_gen.append(child)

        population = next_gen

    print(f"Evolved Autonomous Params: {best_genes}")
    print(f"Out-of-Sample Validation Fitness: {best_val_score:.3f}")

    with open(PARAMS_PATH, "w") as f:
        json.dump(best_genes, f, indent=2)
    return best_genes


# ==========================================
# PART C: AUTONOMOUS LIVE / PAPER EXECUTION
# ==========================================
def get_live_signals(symbols: list[str], model, params: dict) -> dict[str, dict]:
    if not symbols:
        return {}

    now = datetime.now(timezone.utc)
    req = StockBarsRequest(
        symbol_or_symbols=symbols,
        timeframe=TimeFrame(5, TimeFrameUnit.Minute),
        start=now - timedelta(hours=4),
        end=now,
        feed=DataFeed.IEX,
    )
    bars_df = data_client.get_stock_bars(req).df
    if bars_df.empty:
        return {}

    bars_df = bars_df.reset_index()
    signals = {}

    for symbol, group in bars_df.groupby("symbol"):
        if len(group) < 15:
            continue
        feat_df = compute_features(group)
        latest = feat_df.iloc[-1]
        if latest[FEATURES].isna().any():
            continue

        pred_return = float(model.predict(feat_df.iloc[[-1]][FEATURES])[0])
        combined_signal = pred_return + params["vol_weight"] * (float(latest["vol_ratio"]) - 1.0) * 0.001
        signals[symbol] = {
            "signal": combined_signal,
            "pred": pred_return,
            "price": float(latest["close"]),
        }

    return signals


def run_autonomous_bot(model, params: dict):
    account = trading_client.get_account()
    equity = float(account.equity)
    buying_power = float(account.buying_power)
    print(f"Account #{account.account_number} | Status: {account.status.value} | Equity: ${equity:,.2f} | Buying Power: ${buying_power:,.2f}")

    if account.status != AccountStatus.ACTIVE or account.trading_blocked:
        raise RuntimeError(f"Account {account.account_number} cannot trade (status={account.status.value}).")

    clock = trading_client.get_clock()
    if not clock.is_open:
        print("Market is currently closed. Learning complete; skipping order execution.")
        return

    held_positions = trading_client.get_all_positions()
    held_symbols = {p.symbol for p in held_positions}
    all_symbols = list(set(WATCHLIST) | held_symbols)

    signals = get_live_signals(all_symbols, model, params)
    print(f"Evaluated {len(signals)} symbols autonomously.")

    # 1. Autonomous Exits (Stop-loss, Take-profit, or Signal reversal)
    for p in held_positions:
        pl_pct = float(p.unrealized_plpc)
        s = signals.get(p.symbol)

        if pl_pct <= params["stop_loss"]:
            print(f"AUTONOMOUS EXIT [STOP-LOSS] {p.symbol}: {pl_pct * 100:.2f}%")
            trading_client.close_position(p.symbol)
        elif pl_pct >= params["take_profit"]:
            print(f"AUTONOMOUS EXIT [TAKE-PROFIT] {p.symbol}: +{pl_pct * 100:.2f}%")
            trading_client.close_position(p.symbol)
        elif s and s["signal"] <= params["sell_thresh"]:
            print(f"AUTONOMOUS EXIT [SIGNAL] {p.symbol}: Signal {s['signal'] * 100:.4f}%")
            trading_client.close_position(p.symbol)

    # 2. Autonomous Entries (Ranked purely by evolved signal strength, no daily quotas)
    open_orders = trading_client.get_orders(GetOrdersRequest(status=QueryOrderStatus.OPEN))
    excluded = held_symbols | {o.symbol for o in open_orders}

    candidates = sorted(
        [
            (sym, s)
            for sym, s in signals.items()
            if sym not in excluded and s["signal"] >= params["buy_thresh"] and s["price"] > 1.0
        ],
        key=lambda x: x[1]["signal"],
        reverse=True,
    )

    position_budget = equity * params["allocation_pct"]
    orders_placed = 0

    for symbol, s in candidates:
        if buying_power < position_budget or buying_power < s["price"]:
            break

        qty = int(position_budget // s["price"])
        if qty > 0:
            print(
                f"AUTONOMOUS BUY: {symbol} @ ${s['price']:.2f} | "
                f"Signal: {s['signal'] * 100:.4f}% | Qty: {qty} (${qty * s['price']:,.2f})"
            )
            trading_client.submit_order(
                MarketOrderRequest(
                    symbol=symbol,
                    qty=qty,
                    side=OrderSide.BUY,
                    time_in_force=TimeInForce.DAY,
                )
            )
            buying_power -= qty * s["price"]
            orders_placed += 1

    print(f"Cycle complete. Autonomous orders placed: {orders_placed}")


if __name__ == "__main__":
    print("Fetching latest market data for autonomous learning cycle...")
    raw_history = fetch_historical_bars(WATCHLIST[:30], days_back=21)

    ridge_model, trained_df = train_fresh_model(raw_history)
    trained_df["pred_return"] = ridge_model.predict(trained_df[FEATURES])

    autonomous_params = evolve_unconstrained_strategy(trained_df)

    print(f"Connected to Alpaca ({'PAPER' if IS_PAPER else 'LIVE'} mode).")
    run_autonomous_bot(ridge_model, autonomous_params)
