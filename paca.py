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

# 1. Load Credentials (override=True forces .env to replace any cached Windows env vars)
load_dotenv(override=True)

API_KEY = os.getenv("ALPACA_API_KEY")
SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
BASE_URL = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")

if not API_KEY or not SECRET_KEY:
    raise ValueError("Missing ALPACA_API_KEY or ALPACA_SECRET_KEY in .env or GitHub Secrets.")

IS_PAPER = "paper" in BASE_URL.lower()
trading_client = TradingClient(API_KEY, SECRET_KEY, paper=IS_PAPER)
data_client = StockHistoricalDataClient(API_KEY, SECRET_KEY)

# 2. File & Strategy Configuration
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "ridge_model.joblib")
PARAMS_PATH = os.path.join(SCRIPT_DIR, "best_params.json")

FEATURES = ["return_1", "return_3", "hl_spread", "oc_change", "vol_ratio", "ma_dist"]

DAILY_BUY_TARGET = 10      # Target at least 10 buys per trading day
BUDGET_PER_STOCK = 250.0   # $250 per position (~$2,500 deployed across 10 buys)

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
    df["ma_dist"] = (df["close"] - df["close"].rolling(12).mean()) / df["close"]
    return df


def fetch_historical_bars(symbols: list[str], days_back: int = 14) -> pd.DataFrame:
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
# PART B: GENETIC ALGORITHM OPTIMIZER
# ==========================================
def evaluate_fitness(df: pd.DataFrame, genes: dict) -> float:
    buy_thresh = genes["buy_thresh"]
    sell_thresh = genes["sell_thresh"]
    stop_loss = genes["stop_loss"]
    trade_returns = []

    for (_, _), day_df in df.groupby(["symbol", "trade_date"]):
        in_position = False
        entry_price = 0.0
        closes = day_df["close"].values
        preds = day_df["pred_return"].values

        for i in range(len(day_df)):
            price = closes[i]
            pred = preds[i]
            is_last_bar = (i == len(day_df) - 1)

            if not in_position and not is_last_bar:
                if pred >= buy_thresh:
                    in_position = True
                    entry_price = price
            elif in_position:
                current_pl = (price - entry_price) / entry_price
                if current_pl <= stop_loss or pred <= sell_thresh or is_last_bar:
                    trade_returns.append(current_pl - 0.0002)
                    in_position = False

    if len(trade_returns) < 30:
        return -999.0

    returns_arr = np.array(trade_returns)
    return float((np.mean(returns_arr) / (np.std(returns_arr) + 1e-6)) * np.sqrt(len(returns_arr)))


def evolve_parameters(model, pop_size: int = 20, generations: int = 40) -> dict:
    print("Evolving high-frequency parameters via Genetic Algorithm...")
    raw_df = fetch_historical_bars(WATCHLIST[:15], days_back=14)

    chunks = [
        compute_features(g).dropna()
        for (_, _), g in raw_df.groupby(["symbol", "trade_date"])
        if len(g) >= 15
    ]
    clean_df = pd.concat(chunks, ignore_index=True).sort_values("timestamp")
    clean_df["pred_return"] = model.predict(clean_df[FEATURES])

    split = int(len(clean_df) * 0.7)
    train_df, test_df = clean_df.iloc[:split], clean_df.iloc[split:]

    population = [
        {
            "buy_thresh": random.uniform(0.00002, 0.0004),
            "sell_thresh": random.uniform(-0.0005, -0.00002),
            "stop_loss": random.uniform(-0.03, -0.005),
        }
        for _ in range(pop_size)
    ]

    for _ in range(generations):
        scored = sorted([(evaluate_fitness(train_df, ind), ind) for ind in population], key=lambda x: x[0], reverse=True)
        survivors = [ind for _, ind in scored[:4]]
        next_gen = survivors.copy()

        while len(next_gen) < pop_size:
            p1, p2 = random.sample(survivors, 2)
            child = {k: random.choice([p1[k], p2[k]]) for k in p1}
            if random.random() < 0.25:
                child["buy_thresh"] = float(np.clip(child["buy_thresh"] * random.uniform(0.8, 1.2), 0.00001, 0.0005))
            if random.random() < 0.25:
                child["sell_thresh"] = float(np.clip(child["sell_thresh"] * random.uniform(0.8, 1.2), -0.001, -0.00001))
            if random.random() < 0.25:
                child["stop_loss"] = float(np.clip(child["stop_loss"] * random.uniform(0.8, 1.2), -0.05, -0.003))
            next_gen.append(child)

        population = next_gen

    best_train, best_genes = max([(evaluate_fitness(train_df, ind), ind) for ind in population], key=lambda x: x[0])
    out_sample = evaluate_fitness(test_df, best_genes)

    print(f"Evolved Params: {best_genes}")
    print(f"In-Sample Fitness: {best_train:.3f} | Out-of-Sample Fitness: {out_sample:.3f}")

    with open(PARAMS_PATH, "w") as f:
        json.dump(best_genes, f, indent=2)
    return best_genes


# ==========================================
# PART C: ALPACA LIVE / PAPER EXECUTION
# ==========================================
def get_live_predictions(symbols: list[str], model) -> dict[str, tuple[float, float]]:
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
    predictions = {}

    for symbol, group in bars_df.groupby("symbol"):
        if len(group) < 15:
            continue
        feat_df = compute_features(group)
        latest_row = feat_df.iloc[[-1]][FEATURES]
        if latest_row.isna().any().any():
            continue

        pred_return = float(model.predict(latest_row)[0])
        latest_close = float(feat_df.iloc[-1]["close"])
        predictions[symbol] = (pred_return, latest_close)

    return predictions


def count_todays_buys() -> int:
    today_start = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    orders = trading_client.get_orders(
        GetOrdersRequest(status=QueryOrderStatus.ALL, after=today_start)
    )
    return sum(1 for o in orders if o.side == OrderSide.BUY and o.status != "canceled")


def run_alpaca_bot(model, params: dict):
    # Verify account is active before doing anything
    account = trading_client.get_account()
    print(f"Account #{account.account_number} | Status: {account.status.value} | Buying Power: ${float(account.buying_power):,.2f}")

    if account.status != AccountStatus.ACTIVE or account.trading_blocked:
        raise RuntimeError(
            f"Account {account.account_number} cannot trade (status={account.status.value}, blocked={account.trading_blocked}). "
            "Open a new Paper Account in Alpaca and update your .env keys."
        )

    clock = trading_client.get_clock()
    if not clock.is_open:
        print("Market is currently closed. Skipping trade execution.")
        return

    held_positions = trading_client.get_all_positions()
    held_symbols = {p.symbol for p in held_positions}
    all_symbols = list(set(WATCHLIST) | held_symbols)

    predictions = get_live_predictions(all_symbols, model)
    print(f"Scored {len(predictions)} symbols from live 5m bars.")

    # 1. Sell weak positions or negative forecasts
    for p in held_positions:
        pl_pct = float(p.unrealized_plpc)
        pred_return = predictions.get(p.symbol, (0.0, 0.0))[0]

        if pl_pct <= params["stop_loss"]:
            print(f"STOP-LOSS SELL {p.symbol}: Loss of {pl_pct * 100:.2f}%")
            trading_client.close_position(p.symbol)
        elif pred_return <= params["sell_thresh"]:
            print(f"MODEL SELL {p.symbol}: Predicted return {pred_return * 100:.4f}%")
            trading_client.close_position(p.symbol)

    # 2. Check daily buy quota (target at least 10 buys/day)
    buys_today = count_todays_buys()
    remaining_quota = max(0, DAILY_BUY_TARGET - buys_today)
    print(f"Buys placed so far today: {buys_today} | Remaining to hit {DAILY_BUY_TARGET}: {remaining_quota}")

    buying_power = float(account.buying_power)
    if buying_power < BUDGET_PER_STOCK:
        print(f"Insufficient buying power (${buying_power:.2f}).")
        return

    open_orders = trading_client.get_orders(GetOrdersRequest(status=QueryOrderStatus.OPEN))
    excluded = held_symbols | {o.symbol for o in open_orders}

    effective_buy_thresh = min(params["buy_thresh"], 0.0002)

    strong_candidates = sorted(
        [
            (sym, pred, price)
            for sym, (pred, price) in predictions.items()
            if sym not in excluded and pred >= effective_buy_thresh and 5 <= price <= 1000
        ],
        key=lambda x: x[1],
        reverse=True,
    )

    selected = list(strong_candidates)
    if len(selected) < remaining_quota:
        selected_syms = {s[0] for s in selected}
        fallback_candidates = sorted(
            [
                (sym, pred, price)
                for sym, (pred, price) in predictions.items()
                if sym not in excluded and sym not in selected_syms and pred > 0 and 5 <= price <= 1000
            ],
            key=lambda x: x[1],
            reverse=True,
        )
        needed = remaining_quota - len(selected)
        selected.extend(fallback_candidates[:needed])

    max_orders_this_run = max(remaining_quota, min(len(strong_candidates), 15 - buys_today))

    orders_placed = 0
    for symbol, pred_return, price in selected:
        if orders_placed >= max_orders_this_run or buying_power < BUDGET_PER_STOCK:
            break

        qty = int(BUDGET_PER_STOCK // price)
        if qty > 0:
            print(f"BUY SIGNAL [{orders_placed + 1}/{max_orders_this_run}]: {symbol} @ ${price:.2f} | Pred: +{pred_return * 100:.4f}% | Qty: {qty}")
            trading_client.submit_order(
                MarketOrderRequest(
                    symbol=symbol,
                    qty=qty,
                    side=OrderSide.BUY,
                    time_in_force=TimeInForce.DAY,
                )
            )
            buying_power -= qty * price
            orders_placed += 1

    print(f"Run complete. Orders placed this run: {orders_placed} | Total today: {buys_today + orders_placed}")


if __name__ == "__main__":
    ridge_model = get_or_train_model(MODEL_PATH)

    if os.path.exists(PARAMS_PATH):
        with open(PARAMS_PATH, "r") as f:
            active_params = json.load(f)
        print(f"Loaded evolved parameters: {active_params}")
    else:
        active_params = evolve_parameters(ridge_model)

    print(f"Connected to Alpaca ({'PAPER' if IS_PAPER else 'LIVE'} mode).")
    run_alpaca_bot(ridge_model, active_params)