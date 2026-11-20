import os
import json
import random
from datetime import datetime, timedelta, timezone
import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, GetOrdersRequest
from alpaca.trading.enums import OrderSide, TimeInForce, QueryOrderStatus, AccountStatus
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.data.enums import DataFeed

# 1. Load Credentials (strictly anchored to the .env file in this script's folder)
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

# 2. File & Strategy Configuration
MODEL_PATH = os.path.join(SCRIPT_DIR, "ridge_model.joblib")
PARAMS_PATH = os.path.join(SCRIPT_DIR, "best_params.json")

FEATURES = ["return_1", "return_3", "hl_spread", "oc_change", "vol_ratio", "ma_dist"]

DAILY_BUY_TARGET = 10      # Target at least 10 buys per trading day
BUDGET_PER_STOCK = 250.0   # Base allocation per stock (scales 1.5x when averages are strong)

# 60-ticker liquid universe so enough intraday signals fire daily
WATCHLIST = [
    "TSLA", "NVDA", "AAPL", "AMD", "PLTR", "AMZN", "META", "MSFT", "GOOGL", "GOOG",
    "SOFI", "HOOD", "RKLB", "SMCI", "MSTR", "INTC", "MU", "AVGO", "NFLX", "UBER",
    "PYPL", "HIMS", "IONQ", "RIOT", "MARA", "CLSK", "ACHR", "JOBY", "COIN", "SQ",
    "SHOP", "SNOW", "CRWD", "PANW", "NET", "DDOG", "ZS", "CRM", "ORCL", "QCOM",
    "ARM", "TSM", "MRVL", "DELL", "BABA", "NIO", "RIVN", "LCID", "F", "GM",
    "DIS", "BA", "AAL", "DAL", "CCL", "NCLH", "JPM", "BAC", "WFC", "C"
]


# ==========================================
# PART A: DATA FETCHING & FEATURE ENGINEERING
# ==========================================
def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("timestamp").copy()
    df["return_1"] = df["close"].pct_change(1)
    df["return_3"] = df["close"].pct_change(3)
    df["hl_spread"] = (df["high"] - df["low"]) / df["close"]
    df["oc_change"] = (df["close"] - df["open"]) / df["open"]
    df["vol_ratio"] = df["volume"] / df["volume"].rolling(12).mean()

    # Moving average distance (1-hour / 12-bar SMA) and slope
    sma_12 = df["close"].rolling(12).mean()
    df["ma_dist"] = (df["close"] - sma_12) / sma_12
    df["ma_slope"] = sma_12.pct_change(3)

    # Shallow drop metric: pullback from rolling 2-hour (24-bar) high
    rolling_high = df["high"].rolling(24, min_periods=1).max()
    df["drop_from_high"] = (df["close"] - rolling_high) / rolling_high
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


def get_or_train_model(model_path: str):
    if os.path.exists(model_path):
        print(f"Loaded model: {model_path}")
        return joblib.load(model_path)

    print("Fetching 14 days of 5m bars from Alpaca to train Ridge regression model...")
    raw_df = fetch_historical_bars(WATCHLIST[:30], days_back=14)

    chunks = []
    for (_, _), group in raw_df.groupby(["symbol", "trade_date"]):
        if len(group) < 20:
            continue
        feat_df = compute_features(group)
        feat_df["target_next_return"] = feat_df["close"].pct_change(1).shift(-1)
        chunks.append(feat_df.dropna())

    full_df = pd.concat(chunks, ignore_index=True).sort_values("timestamp")
    split_idx = int(len(full_df) * 0.8)
    train_df = full_df.iloc[:split_idx]

    model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    model.fit(train_df[FEATURES], train_df["target_next_return"])

    joblib.dump(model, model_path)
    print(f"Trained on {len(train_df):,} bars and saved model to {model_path}")
    return model


# ==========================================
# PART B: WALK-FORWARD GENETIC ALGORITHM
# ==========================================
def evaluate_fitness(df: pd.DataFrame, genes: dict, target_daily_trades: float = 12.0) -> float:
    ma_buy_thresh = genes["ma_buy_thresh"]
    ma_sell_thresh = genes["ma_sell_thresh"]
    max_drop = genes["max_drop"]
    stop_loss = genes["stop_loss"]

    trade_returns = []
    num_days = df["trade_date"].nunique()
    if num_days == 0:
        return -999.0

    for (_, _), day_df in df.groupby(["symbol", "trade_date"]):
        in_position = False
        entry_price = 0.0

        closes = day_df["close"].values
        preds = day_df["pred_return"].values
        ma_dists = day_df["ma_dist"].values
        ma_slopes = day_df["ma_slope"].values
        drops = day_df["drop_from_high"].values

        for i in range(len(day_df)):
            price = closes[i]
            is_last_bar = (i == len(day_df) - 1)

            if not in_position and not is_last_bar:
                high_average = (ma_dists[i] >= ma_buy_thresh) and (ma_slopes[i] > 0)
                shallow_drop = (drops[i] >= max_drop)
                if high_average and shallow_drop and preds[i] > 0:
                    in_position = True
                    entry_price = price

            elif in_position:
                current_pl = (price - entry_price) / entry_price
                averages_falling = (ma_dists[i] <= ma_sell_thresh) and (ma_slopes[i] < 0)
                if averages_falling or current_pl <= stop_loss or is_last_bar:
                    trade_returns.append(current_pl - 0.0003)
                    in_position = False

    trades_per_day = len(trade_returns) / num_days
    if trades_per_day < 5.0 or trades_per_day > 30.0:
        return -999.0

    returns_arr = np.array(trade_returns)
    sharpe = np.mean(returns_arr) / (np.std(returns_arr) + 1e-6)
    freq_multiplier = 1.0 - (abs(trades_per_day - target_daily_trades) / 30.0)
    return float(sharpe * max(freq_multiplier, 0.2))


def evolve_parameters(model, pop_size: int = 24, generations: int = 35) -> dict:
    print("Evolving moving-average & shallow-drop parameters via Genetic Algorithm...")
    raw_df = fetch_historical_bars(WATCHLIST[:20], days_back=21)

    chunks = [
        compute_features(g).dropna()
        for (_, _), g in raw_df.groupby(["symbol", "trade_date"])
        if len(g) >= 15
    ]
    clean_df = pd.concat(chunks, ignore_index=True).sort_values("timestamp")
    clean_df["pred_return"] = model.predict(clean_df[FEATURES])

    n = len(clean_df)
    train_df = clean_df.iloc[:int(n * 0.6)]
    val_df = clean_df.iloc[int(n * 0.6):int(n * 0.8)]
    test_df = clean_df.iloc[int(n * 0.8):]

    w_size = len(train_df) // 3
    train_windows = [train_df.iloc[i * w_size:(i + 1) * w_size] for i in range(3)]

    def multi_window_fitness(genes: dict) -> float:
        scores = [evaluate_fitness(w, genes) for w in train_windows]
        if any(s <= -900 for s in scores):
            return -999.0
        return float(np.mean(scores) - 0.5 * np.std(scores))

    population = [
        {
            "ma_buy_thresh": random.uniform(0.0005, 0.0060),
            "ma_sell_thresh": random.uniform(-0.0050, 0.0000),
            "max_drop": random.uniform(-0.0150, -0.0010),
            "stop_loss": random.uniform(-0.0250, -0.0060),
        }
        for _ in range(pop_size)
    ]

    best_val_score = -999.0
    best_generalized_genes = population[0]

    for _ in range(generations):
        scored = sorted([(multi_window_fitness(ind), ind) for ind in population], key=lambda x: x[0], reverse=True)
        survivors = [ind for _, ind in scored[:5]]

        val_score = evaluate_fitness(val_df, survivors[0])
        if val_score > best_val_score:
            best_val_score = val_score
            best_generalized_genes = survivors[0].copy()

        next_gen = survivors.copy()
        while len(next_gen) < pop_size:
            p1, p2 = random.sample(survivors, 2)
            child = {k: random.choice([p1[k], p2[k]]) for k in p1}
            if random.random() < 0.25:
                child["ma_buy_thresh"] = float(np.clip(child["ma_buy_thresh"] * random.uniform(0.85, 1.15), 0.0001, 0.01))
            if random.random() < 0.25:
                child["ma_sell_thresh"] = float(np.clip(child["ma_sell_thresh"] * random.uniform(0.85, 1.15), -0.01, 0.001))
            if random.random() < 0.25:
                child["max_drop"] = float(np.clip(child["max_drop"] * random.uniform(0.85, 1.15), -0.03, -0.0005))
            if random.random() < 0.25:
                child["stop_loss"] = float(np.clip(child["stop_loss"] * random.uniform(0.85, 1.15), -0.04, -0.004))
            next_gen.append(child)

        population = next_gen

    final_test_score = evaluate_fitness(test_df, best_generalized_genes)
    print(f"Best Generalized Params: {best_generalized_genes}")
    print(f"Validation Fitness: {best_val_score:.3f} | Unseen Test Fitness: {final_test_score:.3f}")

    with open(PARAMS_PATH, "w") as f:
        json.dump(best_generalized_genes, f, indent=2)
    return best_generalized_genes


# ==========================================
# PART C: ALPACA LIVE / PAPER EXECUTION
# ==========================================
def get_live_metrics(symbols: list[str], model) -> dict[str, dict]:
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
    metrics = {}

    for symbol, group in bars_df.groupby("symbol"):
        if len(group) < 15:
            continue
        feat_df = compute_features(group)
        latest = feat_df.iloc[-1]
        if latest[FEATURES + ["ma_slope", "drop_from_high"]].isna().any():
            continue

        pred_return = float(model.predict(feat_df.iloc[[-1]][FEATURES])[0])
        metrics[symbol] = {
            "pred": pred_return,
            "price": float(latest["close"]),
            "ma_dist": float(latest["ma_dist"]),
            "ma_slope": float(latest["ma_slope"]),
            "drop": float(latest["drop_from_high"]),
        }

    return metrics


def count_todays_buys() -> int:
    today_start = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    orders = trading_client.get_orders(
        GetOrdersRequest(status=QueryOrderStatus.ALL, after=today_start)
    )
    return sum(1 for o in orders if o.side == OrderSide.BUY and o.status != "canceled")


def run_alpaca_bot(model, params: dict):
    account = trading_client.get_account()
    print(f"Account #{account.account_number} | Status: {account.status.value} | Buying Power: ${float(account.buying_power):,.2f}")

    if account.status != AccountStatus.ACTIVE or account.trading_blocked:
        raise RuntimeError(
            f"Account {account.account_number} cannot trade (status={account.status.value}, blocked={account.trading_blocked}). "
            "Check your .env file or GitHub Secrets."
        )

    clock = trading_client.get_clock()
    if not clock.is_open:
        print("Market is currently closed. Skipping trade execution.")
        return

    held_positions = trading_client.get_all_positions()
    held_symbols = {p.symbol for p in held_positions}
    all_symbols = list(set(WATCHLIST) | held_symbols)

    metrics = get_live_metrics(all_symbols, model)
    print(f"Analyzed moving averages & pullbacks for {len(metrics)} symbols.")

    # 1. SELL when averages get lower or stop-loss hits
    for p in held_positions:
        pl_pct = float(p.unrealized_plpc)
        m = metrics.get(p.symbol)

        if pl_pct <= params["stop_loss"]:
            print(f"STOP-LOSS SELL {p.symbol}: Loss of {pl_pct * 100:.2f}%")
            trading_client.close_position(p.symbol)
        elif m and m["ma_dist"] <= params["ma_sell_thresh"] and m["ma_slope"] < 0:
            print(f"AVG DROP SELL {p.symbol}: MA Dist {m['ma_dist'] * 100:.2f}% | Slope {m['ma_slope'] * 100:.3f}%")
            trading_client.close_position(p.symbol)

    # 2. BUY stocks where averages are high AND price drop from session high is shallow
    buys_today = count_todays_buys()
    remaining_quota = max(0, DAILY_BUY_TARGET - buys_today)
    print(f"Buys today: {buys_today} | Remaining to hit {DAILY_BUY_TARGET}: {remaining_quota}")

    buying_power = float(account.buying_power)
    open_orders = trading_client.get_orders(GetOrdersRequest(status=QueryOrderStatus.OPEN))
    excluded = held_symbols | {o.symbol for o in open_orders}

    strong_candidates = sorted(
        [
            (sym, m)
            for sym, m in metrics.items()
            if sym not in excluded
            and m["ma_dist"] >= params["ma_buy_thresh"]
            and m["ma_slope"] > 0
            and m["drop"] >= params["max_drop"]
            and 5 <= m["price"] <= 1000
        ],
        key=lambda x: (x[1]["ma_dist"] + x[1]["drop"]),
        reverse=True,
    )

    selected = list(strong_candidates)
    if len(selected) < remaining_quota:
        selected_syms = {s[0] for s in selected}
        fallback = sorted(
            [
                (sym, m)
                for sym, m in metrics.items()
                if sym not in excluded and sym not in selected_syms and m["ma_slope"] > 0 and 5 <= m["price"] <= 1000
            ],
            key=lambda x: (x[1]["drop"], x[1]["ma_dist"]),
            reverse=True,
        )
        selected.extend(fallback[: remaining_quota - len(selected)])

    max_orders_this_run = max(remaining_quota, min(len(strong_candidates), 15 - buys_today))

    orders_placed = 0
    for symbol, m in selected:
        if orders_placed >= max_orders_this_run or buying_power < BUDGET_PER_STOCK:
            break

        allocation = BUDGET_PER_STOCK * 1.5 if m["ma_dist"] >= (params["ma_buy_thresh"] * 2) else BUDGET_PER_STOCK
        allocation = min(allocation, buying_power)
        qty = int(allocation // m["price"])

        if qty > 0:
            print(
                f"BUY [{orders_placed + 1}/{max_orders_this_run}]: {symbol} @ ${m['price']:.2f} | "
                f"MA Dist: +{m['ma_dist'] * 100:.2f}% | Drop: {m['drop'] * 100:.2f}% | Qty: {qty}"
            )
            trading_client.submit_order(
                MarketOrderRequest(
                    symbol=symbol,
                    qty=qty,
                    side=OrderSide.BUY,
                    time_in_force=TimeInForce.DAY,
                )
            )
            buying_power -= qty * m["price"]
            orders_placed += 1

    print(f"Run complete. Orders placed: {orders_placed} | Total today: {buys_today + orders_placed}")


if __name__ == "__main__":
    ridge_model = get_or_train_model(MODEL_PATH)

    # Re-evolve automatically if best_params.json doesn't exist or uses old parameter keys
    required_keys = {"ma_buy_thresh", "ma_sell_thresh", "max_drop", "stop_loss"}
    if os.path.exists(PARAMS_PATH):
        with open(PARAMS_PATH, "r") as f:
            active_params = json.load(f)
        if not required_keys.issubset(active_params.keys()):
            print("Old parameter schema detected. Re-evolving with moving-average & shallow-drop genes...")
            active_params = evolve_parameters(ridge_model)
        else:
            print(f"Loaded evolved parameters: {active_params}")
    else:
        active_params = evolve_parameters(ridge_model)

    print(f"Connected to Alpaca ({'PAPER' if IS_PAPER else 'LIVE'} mode).")
    run_alpaca_bot(ridge_model, active_params)
