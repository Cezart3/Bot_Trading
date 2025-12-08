"""Backtesting report generation and visualization."""

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional
import json

from backtesting.engine import BacktestResult
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class MonthlyStats:
    """Monthly trading statistics."""

    month: str  # YYYY-MM format
    starting_balance: float = 0.0
    ending_balance: float = 0.0
    pnl: float = 0.0
    return_pct: float = 0.0
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_pct: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    best_trade: float = 0.0
    worst_trade: float = 0.0
    trading_days: int = 0

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "month": self.month,
            "starting_balance": self.starting_balance,
            "ending_balance": self.ending_balance,
            "pnl": self.pnl,
            "return_pct": self.return_pct,
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "win_rate": self.win_rate,
            "profit_factor": self.profit_factor,
            "max_drawdown": self.max_drawdown,
            "max_drawdown_pct": self.max_drawdown_pct,
            "avg_win": self.avg_win,
            "avg_loss": self.avg_loss,
            "best_trade": self.best_trade,
            "worst_trade": self.worst_trade,
            "trading_days": self.trading_days,
        }


class BacktestReport:
    """
    Generate reports from backtest results.

    Supports:
    - Console summary
    - JSON export
    - HTML report with charts
    - Trade list CSV
    """

    def __init__(self, result: BacktestResult, output_dir: str = "data/reports"):
        """Initialize report generator."""
        self.result = result
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._monthly_stats: list[MonthlyStats] = []

    def calculate_monthly_stats(self) -> list[MonthlyStats]:
        """
        Calculate statistics for each month.

        Returns:
            List of MonthlyStats objects.
        """
        if not self.result.trades:
            return []

        # Group trades by month
        trades_by_month = defaultdict(list)
        for trade in self.result.trades:
            month_key = trade.entry_time.strftime("%Y-%m")
            trades_by_month[month_key].append(trade)

        # Sort months
        sorted_months = sorted(trades_by_month.keys())

        monthly_stats = []
        running_balance = self.result.initial_balance

        for month in sorted_months:
            trades = trades_by_month[month]

            # Calculate stats
            starting_balance = running_balance
            month_pnl = sum(t.pnl for t in trades)
            ending_balance = starting_balance + month_pnl

            wins = [t for t in trades if t.pnl > 0]
            losses = [t for t in trades if t.pnl < 0]

            total_wins = sum(t.pnl for t in wins)
            total_losses = abs(sum(t.pnl for t in losses))

            # Calculate max drawdown for the month
            peak = starting_balance
            max_dd = 0
            current_balance = starting_balance

            for trade in sorted(trades, key=lambda t: t.entry_time):
                current_balance += trade.pnl
                if current_balance > peak:
                    peak = current_balance
                dd = peak - current_balance
                if dd > max_dd:
                    max_dd = dd

            # Trading days
            trading_days = len(set(t.entry_time.date() for t in trades))

            stats = MonthlyStats(
                month=month,
                starting_balance=starting_balance,
                ending_balance=ending_balance,
                pnl=month_pnl,
                return_pct=(month_pnl / starting_balance * 100) if starting_balance > 0 else 0,
                total_trades=len(trades),
                winning_trades=len(wins),
                losing_trades=len(losses),
                win_rate=(len(wins) / len(trades) * 100) if trades else 0,
                profit_factor=(total_wins / total_losses) if total_losses > 0 else float('inf'),
                max_drawdown=max_dd,
                max_drawdown_pct=(max_dd / peak * 100) if peak > 0 else 0,
                avg_win=(total_wins / len(wins)) if wins else 0,
                avg_loss=(total_losses / len(losses)) if losses else 0,
                best_trade=max(t.pnl for t in trades) if trades else 0,
                worst_trade=min(t.pnl for t in trades) if trades else 0,
                trading_days=trading_days,
            )

            monthly_stats.append(stats)
            running_balance = ending_balance

        self._monthly_stats = monthly_stats
        return monthly_stats

    def print_monthly_summary(self) -> None:
        """Print monthly statistics to console."""
        if not self._monthly_stats:
            self.calculate_monthly_stats()

        if not self._monthly_stats:
            print("No monthly data available.")
            return

        print("\n" + "=" * 100)
        print("                              MONTHLY PERFORMANCE BREAKDOWN")
        print("=" * 100)

        # Header
        print(f"{'Month':<10} | {'Trades':>7} | {'Win%':>6} | {'P&L':>12} | {'Return':>8} | {'Max DD':>8} | {'PF':>6} | {'Best':>10} | {'Worst':>10}")
        print("-" * 100)

        for stats in self._monthly_stats:
            pnl_str = f"${stats.pnl:+,.0f}"
            ret_str = f"{stats.return_pct:+.1f}%"
            dd_str = f"{stats.max_drawdown_pct:.1f}%"
            pf_str = f"{stats.profit_factor:.2f}" if stats.profit_factor != float('inf') else "INF"
            best_str = f"${stats.best_trade:+,.0f}"
            worst_str = f"${stats.worst_trade:+,.0f}"

            print(
                f"{stats.month:<10} | {stats.total_trades:>7} | {stats.win_rate:>5.1f}% | "
                f"{pnl_str:>12} | {ret_str:>8} | {dd_str:>8} | {pf_str:>6} | "
                f"{best_str:>10} | {worst_str:>10}"
            )

        print("-" * 100)

        # Totals
        total_trades = sum(s.total_trades for s in self._monthly_stats)
        total_wins = sum(s.winning_trades for s in self._monthly_stats)
        total_pnl = sum(s.pnl for s in self._monthly_stats)
        total_return = (self.result.final_balance / self.result.initial_balance - 1) * 100

        print(f"{'TOTAL':<10} | {total_trades:>7} | {(total_wins/total_trades*100) if total_trades else 0:>5.1f}% | "
              f"${total_pnl:>+11,.0f} | {total_return:>+7.1f}% | {self.result.max_drawdown_percent:>7.1f}% | "
              f"{self.result.profit_factor:>6.2f} | "
              f"${max(s.best_trade for s in self._monthly_stats):>+9,.0f} | "
              f"${min(s.worst_trade for s in self._monthly_stats):>+9,.0f}")

        print("=" * 100)

        # Monthly averages
        avg_monthly_return = total_return / len(self._monthly_stats) if self._monthly_stats else 0
        avg_monthly_trades = total_trades / len(self._monthly_stats) if self._monthly_stats else 0
        profitable_months = sum(1 for s in self._monthly_stats if s.pnl > 0)

        print(f"\n  Months Analyzed: {len(self._monthly_stats)}")
        print(f"  Profitable Months: {profitable_months} ({profitable_months/len(self._monthly_stats)*100:.0f}%)")
        print(f"  Average Monthly Return: {avg_monthly_return:.2f}%")
        print(f"  Average Monthly Trades: {avg_monthly_trades:.1f}")
        print("=" * 100 + "\n")

    def print_summary(self) -> None:
        """Print summary to console."""
        r = self.result

        print("\n" + "=" * 70)
        print("                    BACKTEST RESULTS SUMMARY")
        print("=" * 70)

        print(f"\n{'PERFORMANCE METRICS':^70}")
        print("-" * 70)
        print(f"  Initial Balance:      ${r.initial_balance:>15,.2f}")
        print(f"  Final Balance:        ${r.final_balance:>15,.2f}")
        print(f"  Net Profit/Loss:      ${r.total_pnl:>15,.2f} ({((r.final_balance/r.initial_balance)-1)*100:+.1f}%)")
        print(f"  Total Commission:     ${r.total_commission:>15,.2f}")

        print(f"\n{'TRADE STATISTICS':^70}")
        print("-" * 70)
        print(f"  Total Trades:         {r.total_trades:>15}")
        print(f"  Winning Trades:       {r.winning_trades:>15} ({r.win_rate:.1f}%)")
        print(f"  Losing Trades:        {r.losing_trades:>15}")
        print(f"  Average Win:          ${r.average_win:>15,.2f}")
        print(f"  Average Loss:         ${r.average_loss:>15,.2f}")
        print(f"  Profit Factor:        {r.profit_factor:>15.2f}")
        print(f"  Average R-Multiple:   {r.average_rr:>15.2f}")

        print(f"\n{'RISK METRICS':^70}")
        print("-" * 70)
        print(f"  Max Drawdown:         ${r.max_drawdown:>15,.2f}")
        print(f"  Max Drawdown %:       {r.max_drawdown_percent:>15.1f}%")
        print(f"  Sharpe Ratio:         {r.sharpe_ratio:>15.2f}")

        if r.trades:
            print(f"\n{'TRADE BREAKDOWN':^70}")
            print("-" * 70)

            # Group by exit reason
            exit_reasons = {}
            for t in r.trades:
                reason = t.exit_reason or "unknown"
                if reason not in exit_reasons:
                    exit_reasons[reason] = {"count": 0, "pnl": 0}
                exit_reasons[reason]["count"] += 1
                exit_reasons[reason]["pnl"] += t.pnl

            for reason, stats in sorted(exit_reasons.items()):
                print(f"  {reason.upper():20} Count: {stats['count']:>4}  |  PnL: ${stats['pnl']:>10,.2f}")

        print("\n" + "=" * 70)

    def print_trades(self, limit: int = 20) -> None:
        """Print individual trades."""
        if not self.result.trades:
            print("No trades to display.")
            return

        print(f"\n{'TRADE HISTORY (Last ' + str(limit) + ' trades)':^90}")
        print("-" * 90)
        print(f"{'#':>3} | {'Date':^12} | {'Side':^6} | {'Entry':>10} | {'Exit':>10} | {'PnL':>10} | {'Reason':^12}")
        print("-" * 90)

        for i, trade in enumerate(self.result.trades[-limit:], 1):
            date_str = trade.entry_time.strftime("%Y-%m-%d")
            pnl_str = f"${trade.pnl:+,.2f}"
            pnl_color = "" if trade.pnl >= 0 else ""

            print(
                f"{i:>3} | {date_str:^12} | {trade.side.upper():^6} | "
                f"{trade.entry_price:>10.2f} | {trade.exit_price:>10.2f} | "
                f"{pnl_str:>10} | {trade.exit_reason:^12}"
            )

        print("-" * 90)

    def export_json(self, filename: Optional[str] = None) -> str:
        """Export results to JSON."""
        if filename is None:
            filename = f"backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        filepath = self.output_dir / filename

        data = {
            "summary": {
                "initial_balance": self.result.initial_balance,
                "final_balance": self.result.final_balance,
                "total_pnl": self.result.total_pnl,
                "total_trades": self.result.total_trades,
                "winning_trades": self.result.winning_trades,
                "losing_trades": self.result.losing_trades,
                "win_rate": self.result.win_rate,
                "profit_factor": self.result.profit_factor,
                "max_drawdown": self.result.max_drawdown,
                "max_drawdown_percent": self.result.max_drawdown_percent,
                "sharpe_ratio": self.result.sharpe_ratio,
            },
            "trades": [t.to_dict() for t in self.result.trades],
            "equity_curve": [
                {"timestamp": ts.isoformat(), "equity": eq}
                for ts, eq in self.result.equity_curve[::10]  # Sample every 10th point
            ],
        }

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2, default=str)

        logger.info(f"Exported results to {filepath}")
        return str(filepath)

    def export_trades_csv(self, filename: Optional[str] = None) -> str:
        """Export trades to CSV."""
        import csv

        if filename is None:
            filename = f"trades_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

        filepath = self.output_dir / filename

        with open(filepath, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "trade_id", "symbol", "side", "quantity",
                "entry_time", "entry_price", "exit_time", "exit_price",
                "stop_loss", "take_profit", "pnl", "commission",
                "exit_reason", "duration_minutes"
            ])

            for t in self.result.trades:
                writer.writerow([
                    t.trade_id, t.symbol, t.side, t.quantity,
                    t.entry_time.isoformat(), t.entry_price,
                    t.exit_time.isoformat(), t.exit_price,
                    t.stop_loss, t.take_profit, t.pnl, t.commission,
                    t.exit_reason, t.duration_minutes
                ])

        logger.info(f"Exported trades to {filepath}")
        return str(filepath)

    def plot_equity_curve(self, show: bool = True, save: bool = False) -> Optional[str]:
        """Plot equity curve using plotly."""
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
        except ImportError:
            logger.warning("Plotly not installed. Run: pip install plotly")
            return None

        if not self.result.equity_curve:
            logger.warning("No equity data to plot")
            return None

        timestamps = [ts for ts, _ in self.result.equity_curve]
        equities = [eq for _, eq in self.result.equity_curve]

        # Create figure with subplots
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            subplot_titles=("Equity Curve", "Drawdown"),
            row_heights=[0.7, 0.3]
        )

        # Equity curve
        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=equities,
                mode="lines",
                name="Equity",
                line=dict(color="blue", width=1),
            ),
            row=1, col=1
        )

        # Add initial balance line
        fig.add_hline(
            y=self.result.initial_balance,
            line_dash="dash",
            line_color="gray",
            annotation_text="Initial Balance",
            row=1, col=1
        )

        # Drawdown
        peak = self.result.initial_balance
        drawdowns = []
        for eq in equities:
            if eq > peak:
                peak = eq
            dd_pct = ((peak - eq) / peak) * 100
            drawdowns.append(-dd_pct)

        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=drawdowns,
                mode="lines",
                name="Drawdown %",
                line=dict(color="red", width=1),
                fill="tozeroy",
                fillcolor="rgba(255,0,0,0.1)",
            ),
            row=2, col=1
        )

        # Mark trades on equity curve
        for trade in self.result.trades:
            color = "green" if trade.pnl > 0 else "red"
            fig.add_trace(
                go.Scatter(
                    x=[trade.exit_time],
                    y=[equities[min(len(equities)-1, timestamps.index(min(timestamps, key=lambda x: abs(x-trade.exit_time))))] if trade.exit_time in timestamps else self.result.final_balance],
                    mode="markers",
                    marker=dict(color=color, size=8, symbol="circle"),
                    name=f"{'Win' if trade.pnl > 0 else 'Loss'}",
                    showlegend=False,
                    hovertext=f"PnL: ${trade.pnl:.2f}",
                ),
                row=1, col=1
            )

        # Update layout
        fig.update_layout(
            title=f"Backtest Results - {self.result.total_trades} Trades | Win Rate: {self.result.win_rate:.1f}% | PnL: ${self.result.total_pnl:,.2f}",
            height=600,
            showlegend=False,
        )

        fig.update_xaxes(title_text="Date", row=2, col=1)
        fig.update_yaxes(title_text="Equity ($)", row=1, col=1)
        fig.update_yaxes(title_text="Drawdown (%)", row=2, col=1)

        if save:
            filepath = self.output_dir / f"equity_curve_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
            fig.write_html(str(filepath))
            logger.info(f"Saved chart to {filepath}")

        if show:
            fig.show()

        return str(filepath) if save else None

    def generate_html_report(self, include_monthly: bool = True) -> str:
        """Generate comprehensive HTML report with monthly breakdown."""
        r = self.result
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Calculate monthly stats if needed
        if include_monthly and not self._monthly_stats:
            self.calculate_monthly_stats()

        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Backtest Report - ORB + VWAP Strategy (365 Days)</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
        .container {{ max-width: 1400px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        h1 {{ color: #333; border-bottom: 2px solid #007bff; padding-bottom: 10px; }}
        h2 {{ color: #555; margin-top: 30px; }}
        .metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 15px; margin: 20px 0; }}
        .metric {{ background: #f8f9fa; padding: 15px; border-radius: 5px; text-align: center; }}
        .metric-value {{ font-size: 24px; font-weight: bold; color: #007bff; }}
        .metric-label {{ font-size: 12px; color: #666; margin-top: 5px; }}
        .positive {{ color: #28a745 !important; }}
        .negative {{ color: #dc3545 !important; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; font-size: 14px; }}
        th, td {{ padding: 10px; border: 1px solid #ddd; text-align: left; }}
        th {{ background: #343a40; color: white; }}
        tr:nth-child(even) {{ background: #f8f9fa; }}
        tr:hover {{ background: #e9ecef; }}
        .monthly-table th {{ background: #17a2b8; }}
        .summary-row {{ background: #343a40 !important; color: white; font-weight: bold; }}
        .footer {{ margin-top: 30px; text-align: center; color: #666; font-size: 12px; }}
        .stats-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
        .chart-placeholder {{ background: #f0f0f0; height: 300px; display: flex; align-items: center; justify-content: center; border-radius: 8px; margin: 20px 0; }}
        .highlight {{ background: #fff3cd; }}
        .nav {{ background: #343a40; padding: 10px; border-radius: 5px; margin-bottom: 20px; }}
        .nav a {{ color: white; margin-right: 20px; text-decoration: none; }}
        .nav a:hover {{ text-decoration: underline; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="nav">
            <a href="#summary">Summary</a>
            <a href="#monthly">Monthly Breakdown</a>
            <a href="#detailed">Detailed Stats</a>
            <a href="#trades">Trade History</a>
        </div>

        <h1>ORB + VWAP Strategy - 365 Day Backtest Report</h1>
        <p>Generated: {timestamp}</p>
        <p>Period: {self._monthly_stats[0].month if self._monthly_stats else 'N/A'} to {self._monthly_stats[-1].month if self._monthly_stats else 'N/A'}</p>

        <h2 id="summary">Performance Summary</h2>
        <div class="metrics">
            <div class="metric">
                <div class="metric-value">${r.initial_balance:,.0f}</div>
                <div class="metric-label">Initial Balance</div>
            </div>
            <div class="metric">
                <div class="metric-value">${r.final_balance:,.0f}</div>
                <div class="metric-label">Final Balance</div>
            </div>
            <div class="metric">
                <div class="metric-value {'positive' if r.total_pnl >= 0 else 'negative'}">${r.total_pnl:+,.0f}</div>
                <div class="metric-label">Net Profit/Loss</div>
            </div>
            <div class="metric">
                <div class="metric-value {'positive' if r.total_pnl >= 0 else 'negative'}">{((r.final_balance/r.initial_balance)-1)*100:+.1f}%</div>
                <div class="metric-label">Total Return</div>
            </div>
            <div class="metric">
                <div class="metric-value">{r.total_trades}</div>
                <div class="metric-label">Total Trades</div>
            </div>
            <div class="metric">
                <div class="metric-value">{r.win_rate:.1f}%</div>
                <div class="metric-label">Win Rate</div>
            </div>
            <div class="metric">
                <div class="metric-value">{r.profit_factor:.2f}</div>
                <div class="metric-label">Profit Factor</div>
            </div>
            <div class="metric">
                <div class="metric-value negative">{r.max_drawdown_percent:.1f}%</div>
                <div class="metric-label">Max Drawdown</div>
            </div>
        </div>
"""

        # Monthly breakdown section
        if include_monthly and self._monthly_stats:
            profitable_months = sum(1 for s in self._monthly_stats if s.pnl > 0)
            avg_monthly_return = sum(s.return_pct for s in self._monthly_stats) / len(self._monthly_stats)

            html += f"""
        <h2 id="monthly">Monthly Performance Breakdown</h2>
        <div class="metrics">
            <div class="metric">
                <div class="metric-value">{len(self._monthly_stats)}</div>
                <div class="metric-label">Months Analyzed</div>
            </div>
            <div class="metric">
                <div class="metric-value positive">{profitable_months}</div>
                <div class="metric-label">Profitable Months</div>
            </div>
            <div class="metric">
                <div class="metric-value">{profitable_months/len(self._monthly_stats)*100:.0f}%</div>
                <div class="metric-label">Monthly Win Rate</div>
            </div>
            <div class="metric">
                <div class="metric-value">{avg_monthly_return:.2f}%</div>
                <div class="metric-label">Avg Monthly Return</div>
            </div>
        </div>

        <table class="monthly-table">
            <tr>
                <th>Month</th>
                <th>Trades</th>
                <th>Wins</th>
                <th>Win Rate</th>
                <th>P&L</th>
                <th>Return</th>
                <th>Max DD</th>
                <th>Profit Factor</th>
                <th>Avg Win</th>
                <th>Avg Loss</th>
                <th>Best Trade</th>
                <th>Worst Trade</th>
            </tr>
"""
            for stats in self._monthly_stats:
                pnl_class = "positive" if stats.pnl >= 0 else "negative"
                pf_str = f"{stats.profit_factor:.2f}" if stats.profit_factor != float('inf') else "INF"
                html += f"""
            <tr>
                <td><strong>{stats.month}</strong></td>
                <td>{stats.total_trades}</td>
                <td>{stats.winning_trades}</td>
                <td>{stats.win_rate:.1f}%</td>
                <td class="{pnl_class}">${stats.pnl:+,.0f}</td>
                <td class="{pnl_class}">{stats.return_pct:+.2f}%</td>
                <td class="negative">{stats.max_drawdown_pct:.1f}%</td>
                <td>{pf_str}</td>
                <td>${stats.avg_win:,.0f}</td>
                <td>${stats.avg_loss:,.0f}</td>
                <td class="positive">${stats.best_trade:+,.0f}</td>
                <td class="negative">${stats.worst_trade:+,.0f}</td>
            </tr>
"""

            # Summary row
            total_trades = sum(s.total_trades for s in self._monthly_stats)
            total_wins = sum(s.winning_trades for s in self._monthly_stats)
            total_pnl = sum(s.pnl for s in self._monthly_stats)

            html += f"""
            <tr class="summary-row">
                <td><strong>TOTAL</strong></td>
                <td>{total_trades}</td>
                <td>{total_wins}</td>
                <td>{(total_wins/total_trades*100) if total_trades else 0:.1f}%</td>
                <td>${total_pnl:+,.0f}</td>
                <td>{((r.final_balance/r.initial_balance)-1)*100:+.2f}%</td>
                <td>{r.max_drawdown_percent:.1f}%</td>
                <td>{r.profit_factor:.2f}</td>
                <td>${r.average_win:,.0f}</td>
                <td>${r.average_loss:,.0f}</td>
                <td>${max(s.best_trade for s in self._monthly_stats):+,.0f}</td>
                <td>${min(s.worst_trade for s in self._monthly_stats):+,.0f}</td>
            </tr>
        </table>
"""

        html += f"""
        <h2 id="detailed">Detailed Statistics</h2>
        <div class="stats-grid">
            <table>
                <tr><th colspan="2">Performance Metrics</th></tr>
                <tr><td>Initial Balance</td><td>${r.initial_balance:,.2f}</td></tr>
                <tr><td>Final Balance</td><td>${r.final_balance:,.2f}</td></tr>
                <tr><td>Net Profit/Loss</td><td class="{'positive' if r.total_pnl >= 0 else 'negative'}">${r.total_pnl:+,.2f}</td></tr>
                <tr><td>Total Return</td><td class="{'positive' if r.total_pnl >= 0 else 'negative'}">{((r.final_balance/r.initial_balance)-1)*100:+.2f}%</td></tr>
                <tr><td>Total Commission</td><td>${r.total_commission:,.2f}</td></tr>
                <tr><td>Sharpe Ratio</td><td>{r.sharpe_ratio:.2f}</td></tr>
            </table>
            <table>
                <tr><th colspan="2">Trade Statistics</th></tr>
                <tr><td>Total Trades</td><td>{r.total_trades}</td></tr>
                <tr><td>Winning Trades</td><td class="positive">{r.winning_trades}</td></tr>
                <tr><td>Losing Trades</td><td class="negative">{r.losing_trades}</td></tr>
                <tr><td>Win Rate</td><td>{r.win_rate:.1f}%</td></tr>
                <tr><td>Average Win</td><td class="positive">${r.average_win:,.2f}</td></tr>
                <tr><td>Average Loss</td><td class="negative">${r.average_loss:,.2f}</td></tr>
                <tr><td>Profit Factor</td><td>{r.profit_factor:.2f}</td></tr>
                <tr><td>Average R-Multiple</td><td>{r.average_rr:.2f}R</td></tr>
            </table>
        </div>

        <h2>Risk Metrics</h2>
        <div class="metrics">
            <div class="metric">
                <div class="metric-value negative">${r.max_drawdown:,.0f}</div>
                <div class="metric-label">Max Drawdown ($)</div>
            </div>
            <div class="metric">
                <div class="metric-value negative">{r.max_drawdown_percent:.1f}%</div>
                <div class="metric-label">Max Drawdown (%)</div>
            </div>
            <div class="metric">
                <div class="metric-value">{r.sharpe_ratio:.2f}</div>
                <div class="metric-label">Sharpe Ratio</div>
            </div>
            <div class="metric">
                <div class="metric-value">{r.profit_factor:.2f}</div>
                <div class="metric-label">Profit Factor</div>
            </div>
        </div>

        <h2 id="trades">Trade History (Last 50)</h2>
        <table>
            <tr>
                <th>#</th>
                <th>Date</th>
                <th>Side</th>
                <th>Entry</th>
                <th>Exit</th>
                <th>P&L</th>
                <th>Exit Reason</th>
            </tr>
"""

        # Show last 50 trades only in HTML
        for i, t in enumerate(r.trades[-50:], len(r.trades) - 49 if len(r.trades) > 50 else 1):
            pnl_class = "positive" if t.pnl >= 0 else "negative"
            html += f"""
            <tr>
                <td>{i}</td>
                <td>{t.entry_time.strftime('%Y-%m-%d %H:%M')}</td>
                <td>{t.side.upper()}</td>
                <td>{t.entry_price:.2f}</td>
                <td>{t.exit_price:.2f}</td>
                <td class="{pnl_class}">${t.pnl:+,.2f}</td>
                <td>{t.exit_reason}</td>
            </tr>
"""

        html += f"""
        </table>
        <p><em>Showing last 50 of {r.total_trades} trades. Full trade list available in CSV export.</em></p>

        <div class="footer">
            <p>ORB + VWAP Trading Bot - Prop Trading Backtest Report</p>
            <p>Risk Parameters: 4% Daily DD | 10% Account DD | 1% Risk per Trade</p>
        </div>
    </div>
</body>
</html>
"""

        filepath = self.output_dir / f"report_365d_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(html)

        logger.info(f"Generated HTML report: {filepath}")
        return str(filepath)
