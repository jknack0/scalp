"""Fetch historical tick/bar data from Tradovate and replay through pipeline.

Usage:
    python scripts/fetch_tradovate_history.py                    # 2 weeks of 1m bars
    python scripts/fetch_tradovate_history.py --type tick         # 2 weeks of ticks
    python scripts/fetch_tradovate_history.py --type minute --days 5
    python scripts/fetch_tradovate_history.py --replay            # fetch + replay through strategies

Tradovate md/getChart protocol:
    Send: md/getChart\n{id}\n\n{json_params}
    Receive: chart data in chunks, then complete signal

Chart types:
    - Tick: underlyingType="Tick", elementSize=1
    - MinuteBar: underlyingType="MinuteBar", elementSize=1 (or 5, etc.)
"""

import asyncio
import json
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse

import aiohttp
import polars as pl
from dotenv import load_dotenv

load_dotenv()

from src.core.config import BotConfig
from src.core.logging import configure_logging, get_logger
from src.feeds.tradovate import TradovateAuth

configure_logging(log_level="INFO", log_file=None)
logger = get_logger("fetch_history")

# Tradovate MD WebSocket URLs
_DEMO_WS = "wss://md-demo.tradovateapi.com/v1/websocket"
_LIVE_WS = "wss://md.tradovateapi.com/v1/websocket"


async def fetch_chart_data(
    config: BotConfig,
    chart_type: str = "minute",
    days: int = 14,
    element_size: int = 1,
) -> list[dict]:
    """Connect to Tradovate MD WebSocket and fetch historical chart data."""

    auth = TradovateAuth(config)
    await auth.authenticate()
    token = auth.access_token

    ws_url = _DEMO_WS if config.tradovate_demo else _LIVE_WS
    logger.info("connecting", url=ws_url)

    session = aiohttp.ClientSession()
    ws = await session.ws_connect(ws_url)

    # Consume SockJS open frame
    open_frame = await ws.receive(timeout=10)
    if open_frame.type == aiohttp.WSMsgType.TEXT and open_frame.data.strip() == "o":
        logger.info("ws_open")
    else:
        raise ConnectionError(f"Expected 'o' frame, got: {open_frame.data}")

    # Authorize
    auth_msg = f"authorize\n1\n\n{token}"
    await ws.send_str(auth_msg)

    # Wait for auth response
    for _ in range(10):
        resp = await ws.receive(timeout=10)
        if resp.type != aiohttp.WSMsgType.TEXT:
            raise ConnectionError(f"Unexpected type: {resp.type}")
        body = resp.data.strip()
        if body == "h":
            continue
        if '"s":200' in body or '"i":' in body:
            logger.info("authorized")
            break
    else:
        raise ConnectionError("Authorization timed out")

    # Build chart request
    if chart_type == "tick":
        underlying_type = "Tick"
    else:
        underlying_type = "MinuteBar"

    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(days=days)

    chart_params = {
        "symbol": config.symbol,
        "chartDescription": {
            "underlyingType": underlying_type,
            "elementSize": element_size,
            "elementSizeUnit": "UnderlyingUnits",
        },
        "timeRange": {
            "asFarAsTimestamp": start_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        },
    }

    req_msg = f"md/getChart\n2\n\n{json.dumps(chart_params)}"
    logger.info(
        "requesting_chart",
        type=chart_type,
        days=days,
        symbol=config.symbol,
        element_size=element_size,
    )
    await ws.send_str(req_msg)

    # Collect chart data
    all_bars = []
    chunks_received = 0
    done = False

    while not done:
        try:
            msg = await asyncio.wait_for(ws.receive(), timeout=30)
        except asyncio.TimeoutError:
            logger.warning("timeout_waiting_for_data")
            break

        if msg.type != aiohttp.WSMsgType.TEXT:
            if msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                break
            continue

        raw = msg.data.strip()
        if raw == "h" or raw == "[]":
            continue

        # Parse SockJS envelope
        text = raw
        if text.startswith("a"):
            text = text[1:]

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            continue

        if isinstance(data, list):
            for item in data:
                bars, is_done = _process_chart_event(item)
                all_bars.extend(bars)
                if is_done:
                    done = True
        elif isinstance(data, dict):
            bars, is_done = _process_chart_event(data)
            all_bars.extend(bars)
            if is_done:
                done = True

        chunks_received += 1
        if chunks_received % 10 == 0:
            print(f"  Received {len(all_bars)} bars so far...", end="\r")

    # Cleanup
    await ws.close()
    await session.close()
    await auth.close()

    logger.info("chart_data_received", bars=len(all_bars), chunks=chunks_received)
    return all_bars


def _process_chart_event(event: dict) -> tuple[list[dict], bool]:
    """Extract bar data from a chart event. Returns (bars, is_complete)."""
    bars = []
    is_done = False

    # Chart data comes in "d" field with "charts" or "bars" arrays
    if event.get("e") == "chart":
        d = event.get("d", {})

        # Check for end-of-chart signal
        if d.get("eoh", False):
            is_done = True

        # Extract bars
        chart_bars = d.get("bars", [])
        for bar in chart_bars:
            bars.append({
                "timestamp": bar.get("timestamp", ""),
                "open": bar.get("open", 0.0),
                "high": bar.get("high", 0.0),
                "low": bar.get("low", 0.0),
                "close": bar.get("close", 0.0),
                "volume": bar.get("upVolume", 0) + bar.get("downVolume", 0),
                "up_volume": bar.get("upVolume", 0),
                "down_volume": bar.get("downVolume", 0),
                "up_ticks": bar.get("upTicks", 0),
                "down_ticks": bar.get("downTicks", 0),
            })

    # Some responses come as direct chart subscription data
    elif "bars" in event:
        for bar in event["bars"]:
            bars.append({
                "timestamp": bar.get("timestamp", ""),
                "open": bar.get("open", 0.0),
                "high": bar.get("high", 0.0),
                "low": bar.get("low", 0.0),
                "close": bar.get("close", 0.0),
                "volume": bar.get("upVolume", 0) + bar.get("downVolume", 0),
                "up_volume": bar.get("upVolume", 0),
                "down_volume": bar.get("downVolume", 0),
                "up_ticks": bar.get("upTicks", 0),
                "down_ticks": bar.get("downTicks", 0),
            })
        if event.get("eoh", False):
            is_done = True

    # Response to our getChart request
    elif "s" in event and "i" in event:
        # Status response — chart data comes separately
        pass

    return bars, is_done


def bars_to_dataframe(bars: list[dict]) -> pl.DataFrame:
    """Convert raw bar dicts to a Polars DataFrame."""
    if not bars:
        return pl.DataFrame()

    df = pl.DataFrame(bars)

    # Parse timestamps
    if "timestamp" in df.columns:
        df = df.with_columns(
            pl.col("timestamp")
            .str.replace("Z$", "+00:00")
            .str.to_datetime(time_zone="UTC")
            .dt.replace_time_zone(None)
            .alias("timestamp")
        )

    # Sort by time
    df = df.sort("timestamp")

    return df


async def replay_through_pipeline(
    df: pl.DataFrame,
    strategy_names: list[str] | None = None,
) -> None:
    """Replay historical bars through the strategy pipeline."""
    from src.core.events import BarEvent, EventBus, EventType
    from src.core.signal_handler import SignalHandler
    from src.features.feature_hub import FeatureHub
    from src.oms.tradovate_oms import TradovateOMS
    from src.risk.risk_manager import RiskManager

    config = BotConfig.from_yaml()

    bus = EventBus()
    risk = RiskManager(
        max_daily_loss_usd=config.max_daily_loss_usd,
        max_position_contracts=config.max_position_contracts,
        max_signals_per_day=config.max_signals_per_day,
    )

    oms = TradovateOMS(bus, config, paper=True)
    await oms.initialize()

    # Build strategies
    from src.strategies.orb_strategy import ORBConfig, ORBStrategy

    available = {"orb": (ORBConfig, ORBStrategy)}
    if strategy_names is None:
        strategy_names = ["orb"]

    strategies = []
    for name in strategy_names:
        cfg_cls, strat_cls = available[name]
        strategies.append(strat_cls(cfg_cls(), FeatureHub()))

    handler = SignalHandler(bus, strategies, risk, oms)

    print(f"\nReplaying {len(df)} bars through {[s.config.strategy_id for s in strategies]}...")
    print(f"  Date range: {df['timestamp'][0]} → {df['timestamp'][-1]}")

    signals_total = 0
    fills_total = 0

    for row in df.iter_rows(named=True):
        ts = row["timestamp"]
        ts_ns = int(ts.timestamp() * 1_000_000_000) if hasattr(ts, 'timestamp') else 0

        bar = BarEvent(
            symbol=config.symbol,
            open=float(row["open"]),
            high=float(row["high"]),
            low=float(row["low"]),
            close=float(row["close"]),
            volume=int(row["volume"]),
            bar_type="1m",
            timestamp_ns=ts_ns,
        )

        await handler.on_bar(bar)

        # Check for new signals
        for s in strategies:
            if s._signals_today > signals_total:
                signals_total = s._signals_today

    print(f"\n  Results:")
    print(f"    Bars processed: {len(df)}")
    print(f"    Signals generated: {sum(s._signals_today for s in strategies)}")
    print(f"    Position: {oms.position}")
    print(f"    Daily P&L: ${risk.daily_pnl:.2f}")
    print(f"    Orders: {len(oms._orders)}")

    for s in strategies:
        print(f"    [{s.config.strategy_id}] signals={s._signals_today}, bars={s._bars_processed}")


def main():
    parser = argparse.ArgumentParser(description="Fetch Tradovate historical data")
    parser.add_argument("--type", default="minute", choices=["tick", "minute"],
                        help="Chart type (default: minute)")
    parser.add_argument("--days", type=int, default=14,
                        help="Days of history (default: 14)")
    parser.add_argument("--element-size", type=int, default=1,
                        help="Bar size (e.g., 1 for 1-min, 5 for 5-min)")
    parser.add_argument("--output", default="data/tradovate_history",
                        help="Output directory for parquet")
    parser.add_argument("--replay", action="store_true",
                        help="Replay bars through strategy pipeline")
    parser.add_argument("--strategy", nargs="*", default=None,
                        help="Strategies for replay (default: orb)")
    args = parser.parse_args()

    config = BotConfig.from_yaml()
    if not config.tradovate_username:
        print("ERROR: Set TRADOVATE_USERNAME in .env")
        return

    async def run():
        # Fetch data
        bars = await fetch_chart_data(
            config,
            chart_type=args.type,
            days=args.days,
            element_size=args.element_size,
        )

        if not bars:
            print("No data received!")
            return

        df = bars_to_dataframe(bars)
        print(f"\nReceived {len(df)} bars")
        print(f"  Range: {df['timestamp'][0]} → {df['timestamp'][-1]}")
        print(f"  Columns: {df.columns}")

        # Save to parquet
        out_dir = Path(args.output)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / f"{args.type}_{args.element_size}m_{args.days}d.parquet"
        df.write_parquet(str(out_file), compression="zstd")
        print(f"  Saved: {out_file} ({len(df)} rows)")

        # Replay if requested
        if args.replay:
            await replay_through_pipeline(df, args.strategy)

    asyncio.run(run())


if __name__ == "__main__":
    main()
