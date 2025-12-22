import time
import sys
import argparse
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd

# Adaugam radacina proiectului in sys.path
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from brokers.mt5_broker import MT5Broker
from strategies.orb_strategy import ORBStrategy, ORBConfig
from strategies.base_strategy import SignalType
from models.order import Order, OrderType, OrderSide
from config.settings import get_settings
from utils.logger import get_logger, setup_logging
import copy
from datetime import timezone

logger = get_logger("ORB_LIVE")

def run_live_trading(pairs, risk_percent=0.5, dry_run=False):
    print("--- SYSTEM STARTUP: ORB BOT ---")
    setup_logging(level="DEBUG") 
    
    settings = get_settings()
    logger.info(f"--- STARTING ORB LIVE TRADING ---")
    logger.info(f"Pairs: {pairs} | Risk: {risk_percent}% | Mode: {'DRY RUN' if dry_run else 'LIVE'}")
    
    print(f"Connecting to MT5 Broker...")
    broker = MT5Broker(
        login=settings.mt5.login,
        password=settings.mt5.password,
        server=settings.mt5.server,
        path=settings.mt5.path
    )
    
    if not broker.connect():
        print("ERROR: Failed to connect to MT5.")
        return
    
    print("MT5 Connected.")
    
    # Robust Timezone Sync based on Candles
    # We look at the latest M15 candle and compare it to real UTC time
    ref_pair = pairs[0]
    sample_candles = broker.get_candles(ref_pair, "M15", 1)
    if not sample_candles:
        logger.error("Could not fetch candles for time sync. Exit.")
        return
    
    candle_time = sample_candles[0].timestamp
    utc_now = datetime.now(timezone.utc).replace(tzinfo=None)
    
    # Calculate offset rounded to hours
    # If candle is 12:30 and UTC is 08:30, offset is +4
    time_diff = candle_time - utc_now
    offset_hours = round(time_diff.total_seconds() / 3600)
    server_offset = timedelta(hours=offset_hours)
    
    logger.info(f"Sync: Candle Time {candle_time.strftime('%H:%M')} | Real UTC {utc_now.strftime('%H:%M')}")
    logger.info(f"Detected Timezone Offset: {offset_hours} hours. (Chart is GMT{'+' if offset_hours >= 0 else ''}{offset_hours})")

    # 2. Initialize Strategies
    strategies = {}
    last_processed_time = {}

    for pair in pairs:
        pip_size = 0.01 if ("JPY" in pair or "XAU" in pair) else 0.0001
        config = ORBConfig(rr_ratio=2.0, use_adx_filter=True, min_adx=20.0, use_breakeven=True, max_range_pips=150.0, min_sl_pips=10.0)
        strategy = ORBStrategy(symbol=pair, timeframe="M15", config=config, pip_size=pip_size)
        strategy.initialize()
        
        logger.info(f"[{pair}] Pre-calculating today's Asian Range...")
        hist = broker.get_candles(pair, "M15", 80)
        if hist:
            for i in range(len(hist) - 1):
                c_utc = copy.copy(hist[i])
                c_utc.timestamp = hist[i].timestamp - server_offset
                strategy.on_candle(c_utc, hist[:i+1])
            
            if strategy.range_high:
                range_pips = (strategy.range_high - strategy.range_low) / strategy.pip_size
                logger.info(f"[{pair}] RANGE: {strategy.range_low:.5f} - {strategy.range_high:.5f} ({range_pips:.1f} pips)")

        strategies[pair] = strategy
        last_processed_time[pair] = None

    print("Monitoring M15 candles...")
    
    try:
        while True:
            time.sleep(5) # Faster check
            
            for pair, strategy in strategies.items():
                # --- A. CHECK OPEN POSITIONS FOR BREAK-EVEN ---
                open_positions = broker.get_positions(pair)
                for pos in open_positions:
                    bid, ask = broker.get_current_price(pair)
                    current_price = bid if pos.side == "long" else ask
                    
                    strategy.should_exit(pos, current_price)
                    if pos.metadata.get("be_activated") and pos.metadata.get("be_synced") is not True:
                        logger.info(f"[{pair}] Updating Broker SL to Break-Even: {pos.stop_loss}")
                        if broker.modify_position(pos.broker_position_id, stop_loss=pos.stop_loss):
                            pos.metadata["be_synced"] = True

                # --- B. SCAN FOR NEW ENTRIES ---
                candles = broker.get_candles(pair, "M15", 2)
                if len(candles) < 2: continue
                
                closed_candle = candles[0]
                current_open = candles[1]
                
                if last_processed_time[pair] == closed_candle.timestamp:
                    continue
                
                last_processed_time[pair] = closed_candle.timestamp
                
                # Normalize to UTC
                closed_candle_utc = copy.copy(closed_candle)
                closed_candle_utc.timestamp = closed_candle.timestamp - server_offset
                
                logger.info(f"[{pair}] Candle Closed | Server: {closed_candle.timestamp.strftime('%H:%M')} | UTC: {closed_candle_utc.timestamp.strftime('%H:%M')}")
                logger.info(f"[{pair}] Now Watching | Server: {current_open.timestamp.strftime('%H:%M')} (Closes in ~15m)")
                
                history = broker.get_candles(pair, "M15", 50)
                signal = strategy.on_candle(closed_candle_utc, history)
                
                if signal and signal.is_entry:
                    # Risk & Execution... (same as before)

                    logger.info(f"[{pair}] BREAKOUT DETECTED: {signal.signal_type.value} @ {signal.price}")
                    
                    if dry_run:
                        logger.info(f"[{pair}] DRY RUN - Trade skipped.")
                        continue
                    
                    if broker.get_positions(pair):
                        logger.warning(f"[{pair}] Trade skipped - position already open.")
                        continue

                    balance = broker.get_account_balance()
                    risk_amount = balance * (risk_percent / 100.0)
                    sl_pips = abs(signal.price - signal.stop_loss) / strategy.pip_size
                    
                    sym_info = broker.get_symbol_info(pair)
                    pip_val = sym_info.get("pip_value", 10.0)
                    
                    if sl_pips > 0:
                        lots = round(risk_amount / (sl_pips * pip_val), 2)
                        lots = max(sym_info.get("volume_min", 0.01), min(lots, sym_info.get("volume_max", 100.0)))

                        logger.info(f"[{pair}] Executing: {signal.signal_type.value} {lots} lots | SL: {signal.stop_loss:.5f} | TP: {signal.take_profit:.5f}")
                        
                        order = Order(
                            symbol=pair,
                            side=OrderSide.BUY if signal.signal_type == SignalType.LONG else OrderSide.SELL,
                            order_type=OrderType.MARKET,
                            quantity=lots,
                            price=signal.price,
                            stop_loss=signal.stop_loss,
                            take_profit=signal.take_profit,
                            comment="ORB_SNIPER_V1"
                        )
                        
                        if broker.place_order(order):
                            logger.info(f"[{pair}] TRADE SUCCESSFUL.")
                        else:
                            logger.error(f"[{pair}] TRADE FAILED.")
                
    except KeyboardInterrupt:
        print("\nStopping...")
        broker.disconnect()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--pairs', type=str, default='EURUSD,GBPUSD,GBPJPY,USDJPY,XAUUSD')
    parser.add_argument('--risk', type=float, default=0.5)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    run_live_trading([p.strip().upper() for p in args.pairs.split(',')], args.risk, args.dry_run)