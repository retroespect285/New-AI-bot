# New-AI-bot: a self-learning gold (XAUUSD) trading bot for MT5

An automated day-trading bot for **MetaTrader 5**, set up for **gold**. Every 5 minutes it reads
the gold chart on **five timeframes at once (M1, M5, M30, H1, H4)**, forecasts where price is
heading on each of them, decides whether to **BUY**, **SELL** or stay out, and picks its own
**stop-loss**, **take-profit** and holding time. Everything it learns is stored in a **brain file**
that grows every 5 minutes and is reloaded after every restart.

> **Read this first.** This is software, not a money machine. Most automated strategies lose money
> once spreads and commissions are paid, and nothing here guarantees a profit. The bot is built to
> be honest with itself: it only trades setups it has *proven* on recent out-of-sample data, so it
> will often sit on its hands, and on some periods it may find no edge at all. Backtest on your
> broker's data, then run it on a **demo account for weeks** (see [Demo → live](#demo--live)).
> Never risk money you can't afford to lose.

---

## What it does every 5 minutes

```
 new M5 gold candle closes
        │
        ▼
 1. READ     M1 + M5 + M30 + H1 + H4 charts → ~150 indicators and candle patterns
 2. LEARN    every forecast whose answer is now known is scored and learned from (brain file grows)
 3. ACCOUNT  equity, daily-loss / drawdown limits, results of the bot's own closed trades
 4. FORECAST direction outlook for 5 min / 30 min / 1 h / 4 h
 5. DECIDE   score 6 setups: BUY/SELL × 3 trade styles (M5 scalp, M30 intraday, H1 swing) → expected value
 6. MANAGE   open trade: move the stop behind the price, exit if the forecast turns, time limits
 7. EXECUTE  risk checks → lot size from stop distance → market order with SL and TP attached
```

### 1. Reading every timeframe
`tradebot/features.py`. Every indicator is scaled by ATR, so it means the same thing in quiet
and in wild markets.

| chart | what the bot sees there |
|---|---|
| **M5** (the chart it trades) | ~58 indicators: EMA 9–200 distances/slopes/crosses, RSI, MACD, stochastic, ADX, Bollinger bands, 20/50/100-bar support & resistance, momentum over 5 min – 4 h, candle anatomy, engulfing / hammer / shooting star / doji / inside / outside bars, streaks, tick volume, spread, hour of day, weekday |
| **M1** | the small movements *inside* each 5-minute candle: 1-minute momentum, RSI, MACD, wicks, streaks |
| **M30, H1, H4** | the bigger picture: trend (EMA 20/50), momentum, RSI, MACD, ADX, Bollinger position, channel position, last candle's shape and engulfing |
| **M30, H1, H4 still forming** | the candle that hasn't closed yet, rebuilt from M5: "the current H1 candle is up 0.8 ATR, near its high, 40 min old" |

A higher-timeframe candle is only used **after it has closed**, exactly as you would see it live,
so no indicator ever peeks at the future. The tests check this on all charts.

### 2. Predictions on each timeframe
**Direction outlook.** Every 5 minutes the brain forecasts whether gold will be higher in
**5 minutes, 30 minutes, 1 hour and 4 hours**. Each forecast is scored when the answer is known,
which gives a running accuracy per timeframe: the brain's report card. You see it in the log
(`outlook M5:UP52% M30:DN51% H1:UP55% H4:UP57%`: direction and confidence), in
`logs/decisions.csv`, and in `status`.

**Trade styles.** The bot chooses between three ways of trading, each sized by the ATR of its own
chart (all configurable; the rows in italics are available but off by default):

| style | stop-loss | take-profit | max hold | example stop (varies with volatility) |
|---|---|---|---|---|
| M5 scalp | 1.5 × ATR(M5) | 2.0 × ATR(M5) | 1 hour | $3–5 stop |
| M30 intraday | 1.0 × ATR(M30) | 1.5 × ATR(M30) | 4 hours | $5–8 stop |
| H1 intraday swing | 1.0 × ATR(H1) | 2.0 × ATR(H1) | 8 hours | $8–12 stop |
| *M30 trailing* | *1.0 × ATR(M30), then trails 1 ATR behind price* | *3.0 × ATR(M30)* | *4 hours* | *$5–8 stop* |
| *H1 trailing* | *1.0 × ATR(H1), then trails 1 ATR behind price* | *3.0 × ATR(H1)* | *8 hours* | *$8–12 stop* |
| *H4 day trade (optional, off)* | *1.0 × ATR(H4)* | *1.5 × ATR(H4)* | *24 hours* | *$15–25 stop* |

The H4 chart is read for every decision and gets its own 4-hour forecast. Trades *sized* on H4 are
off by default because tests showed they learn poorly: each one takes up to a day to resolve, so
feedback arrives late and consecutive samples are nearly duplicates. You can switch it on in
`config.yaml`.

Each style can be a BUY or a SELL, which gives 6 setups. For each setup the brain estimates
`P(take-profit first)` and `P(stop-loss first)`, then the **expected value** in R (1R = the money
risked): `EV = P(TP)·avgWin + P(SL)·avgLoss + P(timeout)·avgTimeout`, with spread and costs included.
It takes the best setup only if all of these hold:
1. its forecasts have **skill**: they beat simply guessing the base rate, out of sample;
2. **EV ≥ 0.10R** after costs;
3. its **paper-trading record** is good: every signal is also traded on paper, and real orders are
   only allowed while its last 40 trades (paper and real) are profitable on average with a
   t-statistic of at least 1.5.

M1 is used for *reading* only. Trades sized on 1-minute candles would have stops of ~$1, where a
$0.25–0.40 gold spread eats a third of every trade.

### Stop-loss and take-profit that follow the price
The bot sets its own stop-loss and take-profit when it opens a trade, sized to the current
volatility of the chosen style's chart. While the trade is open it keeps managing them by itself;
after every 5-minute candle it:

* **trails the stop** (trailing styles, off by default; see below): once the trade is +1R in
  profit, the stop moves to 1 ATR behind the best price reached, locking in profit. It only moves
  in the trade's favour, never back;
* optionally moves the stop to **break-even** (`risk.breakeven_at_r`);
* **closes early when the forecast turns**: if the brain becomes confident the market will go the
  other way (an opposite setup passes every gate), it closes the trade and may take the opposite one
  (`risk.exit_on_reversal`);
* closes at the style's **max holding time** and before the weekend.

**The brain learns what each exit method is worth.** With trailing styles switched on, every
trailing setup is replayed candle by candle while the bot studies history, with the stop moved
exactly as the live bot moves it. So the
brain knows, for the current chart, whether a fixed target or a trailing stop has the better
expected result, and picks accordingly. A test trades 40 trailing positions through the real
cycle and checks that each result matches what the learning predicted.

**Are trailing stops better? Not on the test data.** With the M30/H1 trailing styles switched on,
the strongly structured test markets returned −1.8% / −0.7% (instead of +5.4% / −3.1% with fixed
exits), and one random walk produced 9 trades where the fixed styles made none. That's why they are
off by default. Gold may behave differently: un-comment the two trailing lines in `config.yaml`, run
the backtest on your exported gold data, and keep whichever version does better.

The stops are moved **on the broker's server** (MT5 "modify position"), so they protect the trade
between candles and even if your PC goes offline. Broker rules are respected: a new stop must keep
the broker's minimum distance from price, and nothing is changed inside the broker's freeze zone. If
price has already come back through the new stop level, the bot closes the trade at market, which is
what the stop would have done.

Why every 5 minutes and not every tick? Updating once per closed candle is what the brain can
learn from and replay exactly. Tick-by-tick trailing would make live behaviour differ from
everything it learned. Between candles, the stop already sits on the server and triggers instantly.

### 3. The brain: how it gets smarter
`tradebot/learner.py`, `tradebot/brain.py`. Each setup and each outlook has its own learner that
blends three "experts":

| expert | what it is | how it adapts |
|---|---|---|
| base rate | recent frequency of the outcome | moving average |
| online logistic regression | linear patterns | updated after **every** new sample |
| gradient-boosted trees | non-linear patterns across charts (e.g. "H4 uptrend + H1 pullback + M5 hammer") | re-trained **daily** on the most recent ~10 days |

The experts are weighted by their recent out-of-sample accuracy, so whichever is reading the
current market best gets the most say. When a pattern stops working, its weight shrinks, the skill
score drops and the paper record turns negative, and the bot stops trading that setup until it
works again.

**It learns from every candle, not just its own trades.** Every 5 minutes, all 6 setups and 4
outlooks whose outcome has become known are scored and learned. That's thousands of lessons per
day, demo or live.

**The brain file** `state/brain_XAUUSD_M5.joblib` holds everything learned: the models, the expert
weights, the average win/loss of every setup, the paper-trading records, the per-timeframe accuracy,
and a **day-by-day learning history**. It is:
* created on the first start by **studying ~70 days of history** on all five charts (a few
  minutes), replayed candle by candle exactly as if it were live;
* updated every 5 minutes, saved every hour and on exit, and reloaded on restart;
* backed up once a day to `state/backups/` (last 7 days kept), so a good brain can be restored.

`python -m tradebot status` shows what is in the brain and whether it is getting smarter:

```
Is it getting smarter? (one row per day; accuracy = recent direction hit rate, best skill = best setup's edge over guessing)
  date            M5    M30     H1     H4   best skill   paper trades
  2026-11-02   50.2%  51.0%  52.4%  53.1%       +0.012            118
  2026-11-03   50.8%  51.6%  53.0%  54.0%       +0.019            131
```
(Illustrative numbers. On real gold, direction accuracy a few points above 50% is already good,
and anyone promising 70–80% is not being honest.)

### 4. Risk management (`tradebot/risk.py`)
* **Position size from the stop distance**: each trade risks `risk_per_trade_pct` (default 0.5%)
  of equity, using the broker's exact gold contract size, tick value and lot step.
* **Adaptive risk**: after a run of losses the risk per trade shrinks (down to 25%) and recovers as results improve.
* **Daily loss limit** (3%), **drawdown kill switch** (15% from the equity peak), **cool-down**
  after 4 losses in a row.
* Spread filter, broker minimum stop distance, one position at a time, trading session (default
  01:00–23:00 server time, which skips gold's spread spike at the daily roll-over), **Friday
  cut-off** (flat before the weekend), stale-data guard.
* **News blackout** (optional): no new trades around NFP / CPI / FOMC (see below).
* **Real-money guard**: the bot refuses a real account unless `mt5.allow_live_account: true`.
* It only ever touches its own trades (`magic` number), never your manual ones.

---

## Installation (Windows + MetaTrader 5)

The official `MetaTrader5` Python package only runs on **Windows**, on the same PC (or Windows
VPS) as the MT5 terminal.

1. Install **MetaTrader 5** from your broker and log in to a **demo account**.
2. In MT5: **Tools → Options → Expert Advisors → tick "Allow algorithmic trading"**, and make sure
   the **Algo Trading** button in the toolbar is green.
3. In MT5: **Tools → Options → Charts → Max bars in chart → Unlimited** (the bot studies history).
4. Show gold in **Market Watch** (right-click → Symbols → search "XAU" or "GOLD"). Note its exact
   name: `XAUUSD`, `GOLD`, `XAUUSD.m`, `XAUUSDm`...
5. Install **Python 3.10–3.13** from python.org (tick *"Add Python to PATH"*).
6. Get this project and run the one-time setup:
   ```bat
   git clone https://github.com/retroespect285/New-AI-bot.git
   cd New-AI-bot
   scripts\install.bat
   ```
   (Or by hand: `python -m venv .venv`, `.venv\Scripts\activate`, `pip install -r requirements.txt`,
   `copy config.example.yaml config.yaml`.)
7. Edit `config.yaml`: set `trading.symbols` to your broker's gold name. If you want the bot to log
   in by itself, copy `.env.example` to `.env` and fill in `MT5_LOGIN`, `MT5_PASSWORD`,
   `MT5_SERVER`. Otherwise it uses whatever account the terminal is logged in to.

## Usage

Run commands from the project folder with the virtual environment active (`.venv\Scripts\activate`).

| step | command | what it does |
|---|---|---|
| 1 | `python -m tradebot check` | tests the connection; shows account (DEMO/REAL), gold spec, spread, and how much M1/M5/M30/H1/H4 history MT5 has |
| 2 | `python -m tradebot export --days 180` | downloads all five charts to `data\XAUUSD_M1.csv` ... `data\XAUUSD_H4.csv` |
| 3 | `python -m tradebot backtest --csv data\XAUUSD_M5.csv` | walk-forward backtest (reads the other four files from the same folder) |
| 4 | `python -m tradebot live --dry-run` | studies history, then follows the live market and logs decisions, **no orders** |
| 5 | `python -m tradebot live` (or `scripts\run_bot.bat`) | trades the demo account. Ctrl+C stops it cleanly |
| – | `python -m tradebot status` | what the brain has learned, day by day |
| – | `python -m tradebot readiness` | does the demo record justify real money? |

**Backtest.** The model starts from zero, learns from the first 6,000 candles without trading, then
trades and keeps learning candle by candle, exactly as it would live (same code). Output in
`backtests\XAUUSD_M5\`: `summary.json`, `equity.csv` (+ `equity.png` if matplotlib is installed),
`orders.csv`, `closed.csv`, `decisions.csv` (why it did or didn't trade on every candle, with the
outlook per timeframe). Without `--csv` it runs on synthetic gold-like data, which only checks
that everything works.

### Files it writes
| path | contents |
|---|---|
| `state/brain_XAUUSD_M5.joblib` | **the brain** (everything learned) |
| `state/backups/` | one brain copy per day, last 7 days |
| `state/bot_state.json` | risk state of the current account (delete it to reset the kill switch) |
| `logs/bot.log` | everything that happened (rotating) |
| `logs/decisions.csv` | every 5 minutes: action, setup, probability, EV, reason, `up_M5`…`up_H4` outlook |
| `logs/orders.csv` | every order sent (or `dry_run`) with SL/TP, lots, money at risk, setup, account |
| `logs/closed.csv` | every closed trade: profit, result in R, setup, account |
| `logs/events.csv` | every stop moved (trail / break-even), reversal exits, time exits, Friday closes |

### News blackout (recommended for gold)
Create e.g. `data\news.csv` with times in your **broker's server time**, and set
`trading.news_file: data/news.csv`:
```
time,title,minutes_before,minutes_after
2026-12-05 15:30,US Non-Farm Payrolls,30,30
2026-12-10 15:30,US CPI,,
2026-12-16 21:00,FOMC rate decision,30,60
```
No new trades are opened inside those windows (blank = the default 15 minutes).

---

## Demo → live

**How long on demo?** The brain does *not* learn faster on demo than live, or the other way
round. It learns from gold prices, which are the same on both. On the first start it already
studies ~70 days of history, far more than one day of demo trading adds. What the demo period has to
prove is that the **whole system makes money**: signals, spreads, fills, your broker, your PC.
That needs **enough trades to tell skill from luck**. One day gives a handful of trades, and a
handful of trades says almost nothing: a coin flip wins 5 in a row about 3% of the time.

So the bot includes a check instead of a calendar:

```bat
python -m tradebot readiness
```
```
Live-trading readiness (based on this bot's demo trades):
  [OK] time on demo: 16.2 days (need 14)
  [  ] number of trades: 21 (need 30)
  [OK] profit factor: 1.41 (need 1.2)
  [OK] average result: +0.18R per trade (need >= +0.00R)
  [OK] worst drawdown: 4.0R ~ 2.0% of equity (limit 10.0%)
NOT READY yet - keep it on demo. Going live now would be betting on luck.
```
The thresholds are in `readiness:` in `config.yaml`. When you do switch:
1. Log the MT5 terminal into the live account (check the gold symbol name is the same; if not,
   rename `state/brain_<old name>_M5.joblib` to the new name so the brain carries over).
2. Set `mt5.allow_live_account: true`, and consider a smaller `risk.risk_per_trade_pct` (e.g. 0.25) at first.
3. Start it. The **brain carries over**; the **risk state starts fresh** for the new account,
   because the demo's equity peak must not trigger the live kill switch. On a real account the
   readiness report is printed at start-up.

## Running it 24/5
* Run it on a **Windows VPS** close to your broker's server, so it doesn't depend on your home PC.
* Use `scripts\run_bot.bat` with Windows Task Scheduler ("At log on") to restart after reboots. It
  resumes from the brain file, and SL/TP stay on the broker's server in the meantime.
* Check `logs/bot.log`, `logs/closed.csv` and `python -m tradebot status` regularly.

## How do I know it isn't fooling itself?
The most common way trading bots go wrong is "learning" patterns that aren't really there.
Safeguards built in:
* **Walk-forward only.** Every forecast is scored on data it had never seen, made at the moment it
  would have been made live. Tests check that every backtest decision is identical whether or not
  future candles exist in the data, on all five charts.
* **Random-walk check.** On pure random-walk prices, where no strategy can win, the bot should not
  trade. During development an early version lost 15% on a random walk; that led to the stop-loss
  model and the paper-trading gate. See *Validation* below for the current results.
* These checks only show that the machinery is honest. **Only a backtest on your broker's real
  gold data, followed by weeks on demo, tells you whether it has an edge.**

## Validation
Walk-forward backtests on artificial gold prices with the default settings (14,000–22,000 M5
candles each, 6,000–8,000 used only for learning before trading starts):

| data | what it should do | result |
|---|---|---|
| **random walk** (4 price series; nothing can be predicted) | stay out | **0 trades** on all 4; direction accuracy 46–55%, i.e. noise around 50% |
| **weak structure** (gold-like series built from M1 with mild trends; 3 series) | trade only if the edge beats costs | 0 trades on all 3; direction accuracy up to 53% where trends exist, not enough after spread |
| **strong structure** (clear trend / mean-reversion regimes; 4 series) | find and trade the edge | +5.4% (21 trades, profit factor 2.8), −3.1% (11 trades), and no trades on 2 series |

What this says: the bot does not invent edges in noise, and it can find and trade a real one, but
even then it is selective and not every period is profitable. Two bugs were found and fixed along
the way:
* an early version **lost 15% on a random walk**; this led to the stop-loss model and the
  paper-trading gate;
* the skill score was **inflated for long timeframes**: the H4 forecast "looked" 55% accurate on
  pure noise until forecasts were frozen at the moment they are made.

None of this predicts results on real gold. Only your broker's data and weeks of demo can do that.

## Limitations (please read)
* **No edge is guaranteed.** The bot is designed to recognise when it has none and then stay out,
  which can mean long stretches without trades.
* Gold spreads and sudden spikes are expensive. Low-spread accounts matter a lot.
* The backtest works from 5-minute candles: inside a candle it assumes the worst (stop before
  target), and it can't simulate slippage, requotes or news spikes exactly.
* News is only handled through the blackout file you maintain.
* One position at a time per symbol; one brain per symbol and trading timeframe.

## Configuration highlights (`config.yaml`)
| setting | default | meaning |
|---|---|---|
| `trading.symbols` | `[XAUUSD]` | your broker's gold symbol name |
| `trading.timeframe` | `M5` | decides once per closed 5-minute candle |
| `model.context_timeframes` | `[M1, M30, H1, H4]` | other charts it reads |
| `model.outlooks` | `[M5, M30, H1, H4]` | direction forecasts it makes and scores |
| `model.styles` | 3 styles (table above) | stop / target / hold / trailing choices |
| `risk.exit_on_reversal` | `true` | close a trade when the brain turns confidently against it |
| `model.min_ev_r` | `0.10` | minimum expected value per trade after costs |
| `risk.risk_per_trade_pct` | `0.5` | % of equity lost when a stop is hit |
| `risk.max_daily_loss_pct` | `3` | daily circuit breaker |
| `risk.max_drawdown_pct` | `15` | kill switch |
| `trading.news_file` | none | news blackout CSV |
| `mt5.allow_live_account` | `false` | must be `true` to trade real money |

See `config.example.yaml` for every option.

## Project layout
```
tradebot/
  features.py    5 charts → ~150 indicators / candle patterns, aligned without look-ahead
  labels.py      what would each setup have done (TP / SL / timeout, in R); direction outcomes
  learner.py     self-adapting probability model (base rate + online logistic + boosted trees)
  brain.py       per-symbol brain: setups, outlooks, EV, skill and paper-trading gates, history
  risk.py        sizing, loss limits, kill switch, adaptive risk
  core.py        the 5-minute cycle shared by live trading and backtests
  engine.py      live loop (waits for each new candle on MT5, fetches all charts)
  backtest.py    walk-forward backtester + statistics
  readiness.py   demo → live readiness report
  news.py        news blackout calendar
  data.py        CSV import/export, resampling, synthetic gold market
  brokers/mt5.py MetaTrader 5 connector (orders, positions, history, filling-mode fallback, reconnect)
  brokers/sim.py candle-replay simulator used by the backtester
  cli.py         command line (check / live / backtest / export / status / readiness)
tests/           unit + integration tests (including "no look-ahead" checks and a fake MT5)
```

## Development
```bash
pip install -r requirements-dev.txt
python -m pytest -q
```
The tests run on any OS: MT5 is replaced by an in-memory fake (`tests/fake_mt5.py`).
