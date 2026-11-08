# New-AI-bot: a self-learning MT5 trading bot

An automated day-trading bot for **MetaTrader 5**. Every time a candle closes (every 5 minutes on
M5) it reads the chart, decides whether to **BUY**, **SELL** or stay out, and picks its own
**stop-loss** and **take-profit**. It keeps learning from the market around the clock and adapts
when conditions change.

> **Read this first.** This is software, not a money machine. Most automated strategies lose money
> once spreads and commissions are paid, and nothing here guarantees a profit. The bot is built to
> be honest with itself: it only trades setups it has *proven* on recent out-of-sample data. It
> will often sit on its hands for days, and on some markets it may never find an edge. Always
> backtest on your broker's data, then run it on a **demo account** for weeks before you even
> consider real money. Never risk money you can't afford to lose.

---

## What it does each bar

```
 new M5 candle closes
        │
        ▼
 1. LEARN    trade setups from 2 hours ago now have a known outcome → they become training data
 2. ACCOUNT  update equity, daily-loss / drawdown limits, learn from the bot's own closed trades
 3. MANAGE   close positions held too long (2h) or before the weekend, optional break-even stop
 4. DECIDE   score every setup: BUY/SELL × 3 stop-loss/take-profit sizes → expected value
 5. EXECUTE  risk checks → lot size from stop distance → market order with SL and TP attached
```

### 1. Reading the chart
`tradebot/features.py` turns raw candles into ~58 indicators that describe the chart the way a
trader would. All of them are scaled by ATR, so they mean the same thing in quiet and in wild markets:

* **trend**: distance from EMA 9/21/50/100/200, EMA crosses and slopes, ADX, +DI/−DI
* **momentum**: price change over 1–48 bars, RSI 7/14, MACD, stochastic
* **volatility**: ATR regime, Bollinger position and width, candle range
* **support/resistance**: position inside the 20/50/100-bar channel, distance to recent highs and lows
* **candlestick patterns**: body and wicks, engulfing, hammer, shooting star, doji, inside/outside bars, streaks
* **context**: tick volume, spread, hour of day, day of week

A feature at bar *i* only uses bars up to *i* (no peeking at the future). The tests check this.

### 2. Choosing direction, stop-loss and take-profit
The bot thinks in **setups** ("arms"): `BUY` or `SELL` combined with a stop-loss/take-profit pair
measured in ATR. The defaults are `1.0/1.5`, `1.5/2.5` and `2.0/4.0` ATR, which makes 6 setups.
For each one it estimates:

* `P(TP)`: the chance the take-profit is hit first
* `P(SL)`: the chance the stop-loss is hit first (if neither is hit, the trade times out after 2 hours)
* **Expected value** in R (1R = the money risked):
  `EV = P(TP)·avgWin + P(SL)·avgLoss + P(timeout)·avgTimeout`, with the averages measured from the
  market itself, spread and costs included

It takes the setup with the highest EV, but only if all of these hold:
1. The model has **skill**: its out-of-sample forecasts beat simply guessing the base rate.
2. **EV ≥ 0.10R** after costs.
3. The setup's **paper-trading record** is good: every signal is also traded on paper in the
   background, and real orders are only allowed while its last 40 trades (paper and real) are
   profitable on average, with a t-statistic of at least 1.5.

SL and TP are sent with the order, so they sit on the broker's server. If your PC or internet
dies, the position is still protected.

### 3. Learning and adapting
`tradebot/learner.py` and `tradebot/brain.py`. Each setup has two forecasting models, and each
model blends three "experts":

| expert | what it is | how it adapts |
|---|---|---|
| base rate | recent frequency of the outcome | moving average |
| online logistic regression | linear patterns | updated after **every** new sample |
| gradient-boosted trees | non-linear patterns (e.g. "breakout + high volume + London open") | re-trained **daily** on the most recent ~10 days |

The experts are blended with **Hedge weights** based on each one's recent out-of-sample accuracy.
Whichever is reading the current market best gets the most say. When a pattern stops working, its
accuracy drops, its weight shrinks, the skill score falls, the paper record turns negative, and the
bot stops trading that setup until it works again.

It learns from **every bar** (would a trade opened here have worked?), not only from its own
trades, so it gathers ~288 lessons per day per setup on M5. Models are saved to `state/` and
picked up again after a restart.

### 4. Risk management (`tradebot/risk.py`)
* **Position size from the stop distance**: each trade risks `risk_per_trade_pct` (default 0.5%)
  of equity, using the broker's exact tick value and lot step.
* **Adaptive risk**: after a run of losses the risk per trade shrinks (down to 25%) and recovers as results improve.
* **Daily loss limit** (3%): no new trades for the rest of the server day.
* **Drawdown kill switch** (15% from the equity peak): trading stops until you reset it by hand.
* **Cool-down** after 4 losses in a row (60 min).
* Spread filter, broker minimum stop distance, max positions per symbol and in total, session
  hours, **Friday cut-off** (flat before the weekend), stale-data guard.
* **Real-money guard**: the bot refuses to trade a real account unless you set `mt5.allow_live_account: true`.
* `magic` number: the bot only ever touches its own trades, never your manual ones.

---

## Installation (Windows + MetaTrader 5)

The official `MetaTrader5` Python package only runs on **Windows**, on the same PC (or Windows
VPS) as the MT5 terminal.

1. Install **MetaTrader 5** from your broker and log in. Open a **demo account** first.
2. In MT5: **Tools → Options → Expert Advisors → tick "Allow algorithmic trading"**, and make sure
   the **Algo Trading** button in the toolbar is green.
3. In MT5: **Tools → Options → Charts → Max bars in chart → Unlimited** (the bot needs history to learn).
4. Install **Python 3.10–3.13** from python.org (tick *"Add Python to PATH"*).
5. Get this project and run the one-time setup:
   ```bat
   git clone https://github.com/retroespect285/New-AI-bot.git
   cd New-AI-bot
   scripts\install.bat
   ```
   (Or by hand: `python -m venv .venv`, `.venv\Scripts\activate`, `pip install -r requirements.txt`,
   `copy config.example.yaml config.yaml`.)
6. Edit `config.yaml`: at least `trading.symbols`, using the exact names from MT5's Market Watch
   (some brokers use suffixes like `EURUSD.m`). If you want the bot to log in by itself, copy
   `.env.example` to `.env` and fill in `MT5_LOGIN`, `MT5_PASSWORD`, `MT5_SERVER`. Otherwise it
   uses whatever account the terminal is logged in to.

## Usage

Run commands from the project folder with the virtual environment active (`.venv\Scripts\activate`).

### Step 1: check the connection
```bat
python -m tradebot check
```
This prints your account (DEMO/REAL), each symbol's contract spec, the spread and how much history is available.

### Step 2: backtest on your broker's data
```bat
python -m tradebot export --bars 100000
python -m tradebot backtest --csv data\EURUSD_M5.csv
```
The backtest is **walk-forward**. The model starts from zero, learns from the first 6,000 bars
without trading, then trades and keeps learning bar by bar, exactly as it would live (it runs the
same code). Output in `backtests\EURUSD_M5\`: `summary.json`, `equity.csv` (+ `equity.png` if
matplotlib is installed), `orders.csv`, `closed.csv`, `decisions.csv` (why it did or didn't trade
on every bar).

You can also use a file from MT5's own *View → Symbols → Bars → Export*. Without `--csv` it
runs on synthetic data, which only checks that everything works and says nothing about real markets.

### Step 3: dry run on live data
```bat
python -m tradebot live --dry-run
```
Learns from history, then follows the live market and **logs** what it would do without sending orders.

### Step 4: demo trading
```bat
python -m tradebot live          (or double-click scripts\run_bot.bat)
```
Keep the window open; **Ctrl+C** stops it cleanly. The first start learns from ~70 days of
history, which takes a few minutes. Later starts resume from `state/`. Let it run on demo for
several weeks and compare with the backtest.

### Inspecting what it learned
```bat
python -m tradebot status
```
Per setup this shows: samples learned, skill vs. the base rate, expert weights, and the paper-trading record.

### Files it writes
| path | contents |
|---|---|
| `logs/bot.log` | everything that happened (rotating) |
| `logs/decisions.csv` | every bar: action, probability, EV, reason |
| `logs/orders.csv` | every order sent (or `dry_run`) with SL/TP, lots and money at risk |
| `logs/closed.csv` | every closed trade with profit and result in R |
| `logs/events.csv` | time exits, Friday closes, break-even moves |
| `state/` | saved models and risk state (delete `bot_state.json` to reset the kill switch) |

## Configuration highlights (`config.yaml`)
| setting | default | meaning |
|---|---|---|
| `trading.symbols` | `[EURUSD]` | one model per symbol |
| `trading.timeframe` | `M5` | acts once per closed candle |
| `trading.commission_per_lot` | `0` | set it if your account charges commission (e.g. 7.0), so learning includes it |
| `model.exit_profiles` | 1/1.5, 1.5/2.5, 2/4 ATR | candidate stop-loss/take-profit sizes the bot chooses from |
| `model.horizon_bars` | `24` | max holding time (2 h on M5) |
| `model.min_ev_r` | `0.10` | minimum expected value per trade after costs |
| `risk.risk_per_trade_pct` | `0.5` | % of equity lost when a stop is hit |
| `risk.max_daily_loss_pct` | `3` | daily circuit breaker |
| `risk.max_drawdown_pct` | `15` | kill switch |
| `mt5.allow_live_account` | `false` | must be `true` to trade real money |

See `config.example.yaml` for every option.

## Running it 24/5
* Run it on a **Windows VPS** close to your broker's server, so it doesn't depend on your home PC.
* Use `scripts\run_bot.bat` with Windows Task Scheduler ("At log on") to restart after reboots. It
  resumes from `state/`, and SL/TP stay on the server in the meantime.
* Check `logs/bot.log` and `logs/closed.csv` regularly. The `status` command shows whether the
  model still has an edge.

## How do I know it isn't fooling itself?
The most common way trading bots go wrong is "learning" patterns that aren't really there.
Safeguards built in:
* **Walk-forward only.** The model is scored on forecasts made *before* each outcome was known,
  exactly as in live trading. A test checks that every backtest decision is identical whether or
  not future bars exist in the data.
* **Random-walk check.** On pure random-walk prices (where no strategy can win), the bot should
  not trade. During development, an early version lost 15% on a random walk; that led to the
  stop-loss probability model and the paper-trading gate. With the current defaults it made
  **0 trades on 4 different random walks**. On synthetic data that does contain trends and
  mean-reversion, it traded selectively (for example 53 trades, profit factor 2.0, max drawdown
  4.3% on one series) and stayed out of others where it couldn't confirm an edge.
* These checks only show that the machinery is honest. **Only a backtest on your broker's real
  data, followed by weeks on demo, tells you whether it has an edge on your market.**

## Limitations (please read)
* **No edge is guaranteed.** The bot is designed to recognise when it has none and then stay out,
  which can mean long stretches without trades.
* Spreads and commissions are a big hurdle on M5. A 1-pip spread against a 5-pip stop costs 0.2R
  per trade before anything else. Low-spread symbols and accounts matter a lot.
* The backtest works from candles: inside a candle it assumes the worst (stop before target),
  and it can't simulate slippage, requotes or news spikes exactly.
* It knows nothing about news or economic releases. Consider widening `session` or pausing it around major events.
* One model per symbol and timeframe; positions are capped at one per symbol.

## Project layout
```
tradebot/
  features.py    chart → indicators / candlestick patterns
  labels.py      what would a trade opened at each bar have done (TP / SL / timeout, in R)
  learner.py     self-adapting probability model (base rate + online logistic + boosted trees, Hedge-blended)
  brain.py       per-symbol decision maker: setups, EV, skill gate, paper-trading gate
  risk.py        sizing, loss limits, kill switch, adaptive risk
  core.py        the per-bar cycle shared by live trading and backtests
  engine.py      live loop (waits for each new candle on MT5)
  backtest.py    walk-forward backtester + statistics
  brokers/mt5.py MetaTrader 5 connector (orders, positions, history, filling-mode fallback, reconnect)
  brokers/sim.py bar-replay simulator used by the backtester
  cli.py         command line (check / live / backtest / export / status)
tests/           unit + integration tests (including "no look-ahead" checks and a fake MT5)
```

## Development
```bash
pip install -r requirements-dev.txt
python -m pytest -q
```
The tests run on any OS: MT5 is replaced by an in-memory fake (`tests/fake_mt5.py`).
