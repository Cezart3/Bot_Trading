# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

**IMPORTANT: La fiecare conversație nouă, citește acest fișier PRIMUL pentru a înțelege starea curentă a proiectului și ce trebuie implementat.**

---

## INSTRUCȚIUNI PENTRU CLAUDE

### La Începutul Fiecărei Conversații:
1. **CITEȘTE ACEST FIȘIER** - conține starea curentă a proiectului
2. Verifică ce faze sunt completate și ce trebuie făcut
3. Continuă de unde ai rămas
4. După modificări majore, actualizează acest fișier
5. Rulează backtest pentru a valida schimbările

---

## STATUS CURENT AL PROIECTULUI

### Ultima Actualizare: 2025-12-17

### Strategie Activă: SMC V3 (OPTIMIZAT PER-SIMBOL)

**Fișiere principale:**
- `strategies/smc_strategy_v3.py` - Strategia completă cu logging detaliat
- `scripts/run_smc_v3.py` - Script pentru rulare live/demo (heartbeat + per-symbol config)
- `backtests/backtest_multi_instrument.py` - Backtest

---

## CUM SA RULEZI BOTUL

### Recomandat: GBPUSD (cel mai profitabil)
```bash
cd D:\Proiecte\BOT_TRADING\Bot_Trading
python scripts/run_smc_v3.py --mode demo --symbols GBPUSD --risk 0.5 --max-trades 4 --log-level DEBUG
```

### Toate simbolurile optimizate
```bash
python scripts/run_smc_v3.py --mode demo --risk 0.5 --max-trades 4 --log-level DEBUG
```

### Paper Trading (fără ordine reale)
```bash
python scripts/run_smc_v3.py --mode paper --symbols GBPUSD AUDUSD --risk 0.5
```

### Parametri Disponibili
- `--mode`: demo (default), paper, live
- `--symbols`: Lista de simboluri (default: GBPUSD AUDUSD EURUSD US30 USTEC GER40)
- `--risk`: Risk % per trade (default: 0.5)
- `--max-trades`: Max trades pe zi (default: 2)
- `--log-level`: DEBUG (recomandat pentru monitorizare), INFO, WARNING, ERROR

---

## SIMBOLURI SUPORTATE (OPTIMIZATE PE BAZA BACKTEST)

### Priorități Bazate pe Performanță Reală:
| Simbol | Prioritate | Backtest P&L | Win Rate | Setări |
|--------|------------|--------------|----------|--------|
| **GBPUSD** | 1 (BEST) | +$784.91 | 50% | RELAXATE |
| AUDUSD | 2 | +$76.07 | 35% | MODERATE |
| EURUSD | 3 | -$177.60 | 29% | **STRICTE** |
| EURGBP | 4 | N/A | N/A | STRICTE |
| US30 | 5 | -$324.02 | 17% | **FOARTE STRICTE** |
| USTEC | 6 | N/A | N/A | STRICTE |
| GER40 | 7 | N/A | N/A | STRICTE |

### Configurație Per-Simbol:
```python
# GBPUSD (BEST) - Setări relaxate
poi_min_score = 1.5
require_sweep = False
adx_trending = 20.0
min_rr = 1.5

# EURUSD (LOSING) - Setări stricte
poi_min_score = 2.5
require_sweep = True  # OBLIGATORIU liquidity sweep!
adx_trending = 25.0
min_rr = 2.0

# US30 (VERY BAD) - Setări foarte stricte
poi_min_score = 3.0
require_sweep = True
adx_trending = 28.0
min_rr = 2.5
```

### Sesiuni de Trading (UTC / Romania)
- **London Kill Zone:** 08:00-11:00 UTC (10:00-13:00 Romania)
- **NY Kill Zone:** 14:00-17:00 UTC (16:00-19:00 Romania)
- **NY Opens:** 14:30 UTC = 16:30 Romania

---

## TP DINAMIC BAZAT PE LICHIDITATE (NOU!)

### Concept
În loc de TP fix bazat pe R:R, strategia acum:
1. **Caută cel mai apropiat punct de lichiditate neatins**
2. **Calculează R:R real** până la acel punct
3. **Intră DOAR dacă R:R >= min_rr** (1.5 pentru GBPUSD, 2.0+ pentru altele)
4. **NU ARE FALLBACK** - dacă nu e target valid, NU intră!

### Ținte de Lichiditate (în ordine de prioritate)
| Tip | Descriere | Prioritate |
|-----|-----------|------------|
| **EQH/EQL** | Equal Highs/Lows (cele mai multe stop-uri) | 1.0+ |
| **PDH/PDL** | Previous Day High/Low | 0.95 |
| **PWH/PWL** | Previous Week High/Low | 0.90 |
| **Session H/L** | Session High/Low (azi) | 0.80 |
| **Swing H/L** | Swing Highs/Lows | 0.70 |

### Exemplu
```
Entry LONG la 1.2650
SL la 1.2635 (15 pips)
Cel mai apropiat EQH la 1.2680 (30 pips distanță)
R:R = 30/15 = 2.0 ✓ INTRĂ

Entry LONG la 1.2650
SL la 1.2635 (15 pips)
Cel mai apropiat target la 1.2665 (15 pips distanță)
R:R = 15/15 = 1.0 ✗ NU INTRĂ (sub min_rr 1.5)
```

### Log-uri TP Dinamic
```
SMC [GBPUSD] TP Target: PDH at 1.26800, R:R=2.00
SMC [GBPUSD] Skip: Closest liquidity EQH at 1.26600 gives R:R=1.20, need >= 1.5
SMC [EURUSD] Skip: No liquidity targets found
```

---

## LOGGING ȘI MONITORIZARE

### Heartbeat (la fiecare 5 minute)
```
========== HEARTBEAT 14:30:00 UTC ==========
Session: ny | Scans: 1800 | Signals: 2
Balance: $5,000.00 | Equity: $5,000.00 | Positions: 0
Trades today: 0/4
==================================================
```

### Debug Logging (când nu găsește semnale)
```
[GBPUSD] Skip: Neutral bias (H1=neutral, EMA=bullish)
[GBPUSD] Skip: No valid POIs found (OBs=3, bias=bullish)
[GBPUSD] Skip: Price not in POI zone (POIs=2, price=1.26500)
[GBPUSD] Skip: No confirmation pattern for long at POI
[GBPUSD] Skip: Ranging market (ADX=15.2 < 20)
```

---

## CONFIGURAȚIE RISK MANAGEMENT

```python
# Risk procentual - NU hardcodat!
risk_percent = 0.5  # 0.5% din cont per trade
max_trades_per_day = 4  # Crescut de la 2

# Equity curve trading
# Dupa 2 losses consecutive -> risk se reduce la 0.25%
# Dupa 3 wins consecutive -> risk revine la normal
```

### Limits Prop Trading
- Max Daily Drawdown: 4%
- Max Account Drawdown: 10%
- Warning la 2% daily DD -> risk se reduce automat

---

## REZULTATE BACKTEST (DATE REALE)

| Symbol | Trades | Win% | PF | P&L |
|--------|--------|------|-------|------|
| GBPUSD | 34 | 50% | 1.90 | +$784.91 |
| AUDUSD | 23 | 35% | 1.10 | +$76.07 |
| EURUSD | 31 | 29% | 0.84 | -$177.60 |
| US30 | 12 | 17% | 0.34 | -$324.02 |
| **TOTAL** | 100 | - | - | **+$359.36** |

**Ținta:** 2-4% pe lună (~$100-200 pe $5000)

---

## FAZE IMPLEMENTATE

### TOATE CELE 5 FAZE - COMPLETATE + OPTIMIZĂRI

- [x] **Faza 1:** Displacement, Liquidity Sweep, Kill Zones
- [x] **Faza 2:** ADX Filter, Volatility Filter, Session Quality
- [x] **Faza 3:** Multi-Candle Confirm, OB Refinement, POI Freshness
- [x] **Faza 4:** Partial TP, Trailing Stop, Time Exit
- [x] **Faza 5:** HTF Confluence, Previous Day Levels, Equity Curve
- [x] **Faza 6:** Per-Symbol Optimization (STRICT/RELAXED settings)
- [x] **Faza 7:** Detailed Logging & Heartbeat

---

## AUTO-DETECTION SIMBOLURI

Botul încearcă automat aliasuri pentru simbolurile index:
```python
SYMBOL_ALIASES = {
    "NAS100": ["USTEC", "USTech100", "NASDAQ"],
    "GER30": ["GER40", "DE40", "DAX"],
    "US30": ["DJI30", "DOW30", "DJ30"],
}
```

---

## TROUBLESHOOTING

### Bot nu afișează loguri
1. Folosește `--log-level DEBUG` pentru logging detaliat
2. Heartbeat apare la fiecare 5 minute când rulează
3. În timpul orelor de non-trading, vezi mesaj "Outside trading hours"

### No trades generated
1. **Check ADX** - sub valorile per-simbol = ranging, no trades
   - GBPUSD: ADX < 20 = skip
   - EURUSD: ADX < 25 = skip
   - US30: ADX < 28 = skip
2. **Check Sweep** - EURUSD și US30 NECESITĂ liquidity sweep
3. **Check POI Score** - fiecare simbol are prag diferit

### Simboluri index nu funcționează
1. Verifică numele exact în MT5 (poate fi USTEC în loc de NAS100)
2. Botul încearcă automat aliasurile
3. Check log pentru "Using X instead of Y"

---

## NEXT STEPS / TODO (Sesiunea Următoare)

### Prioritate 1: Rebalansare Setări Per-Simbol
După implementarea TP dinamic, EURUSD a devenit excelent (PF 5.82!) dar GBPUSD și AUDUSD au scăzut.

**GBPUSD** (era PF 1.69 → acum PF 1.15):
- [ ] Crește `min_rr` de la 1.5 la **1.8** (acceptă mai multe target-uri valide)
- [ ] Crește `poi_min_score` de la 1.5 la **1.8** (calitate mai bună)
- [ ] Păstrează `require_sweep=False`

**AUDUSD** (era PF 0.72 → acum PF 0.46):
- [ ] Activează `require_sweep=True` (la fel ca EURUSD care merge bine)
- [ ] Crește `poi_min_score` de la 1.8 la **2.2**
- [ ] Păstrează `min_rr=1.8`

**EURUSD** (era PF 0.80 → acum PF 5.82):
- [ ] **NU MODIFICA** - funcționează excelent cu setările actuale!

**US30** (0 trades):
- [ ] Reduce `poi_min_score` de la 3.0 la **2.5**
- [ ] Reduce `adx_trending` de la 28 la **25**

### Prioritate 2: Validare Live Demo
- [ ] Rulează botul pe demo 2-3 zile cu noile setări
- [ ] Verifică că generează semnale și loguri corecte
- [ ] Monitorizează heartbeat la fiecare 5 minute

### Prioritate 3: Ajustări Fine
- [ ] Dacă un simbol continuă să piardă după rebalansare, exclude-l temporar
- [ ] Focus pe EURUSD + GBPUSD inițial (cele mai lichide)

---

## CHANGELOG

### 2025-12-17: TP DINAMIC BAZAT PE LICHIDITATE
- **ELIMINAT fallback la fixed R:R** - acum intră DOAR cu target de lichiditate valid
- TP = cel mai apropiat punct de lichiditate neatins (EQH/EQL, PDH/PDL, PWH/PWL)
- Verifică R:R real până la target >= min_rr înainte de intrare
- Adăugat Previous Week High/Low (PWH/PWL) ca target
- Adăugat Session High/Low ca target
- Log detaliat: ce target a găsit și de ce a respins trade-ul

### 2025-12-17: Per-Symbol Optimization & Logging
- **GBPUSD acum prioritate #1** (cel mai profitabil)
- Setări STRICTE pentru EURUSD (require sweep, POI 2.5, ADX 25)
- Setări FOARTE STRICTE pentru US30 (POI 3.0, ADX 28, RR 2.5)
- **Heartbeat logging** la fiecare 5 minute
- **Debug logging** detaliat pentru fiecare skip reason
- Auto-detection pentru simboluri index (aliasuri)
- Scan counter și signals counter

### 2024-12-17: Multi-Symbol Support
- Creat `run_smc_v3.py` pentru rulare cu multiple simboluri
- Risk procentual (0.5% default)

### 2024-12-17: SMC V3 Implementation
- Implementat toate cele 5 faze
- Backtest results: GBPUSD best performer

---

## NOTĂ FINALĂ

**Pentru a porni botul ACUM (recomandat):**

```bash
cd D:\Proiecte\BOT_TRADING\Bot_Trading
python scripts/run_smc_v3.py --mode demo --risk 0.5 --max-trades 4 --log-level DEBUG
```

**Acesta va:**
1. Conecta la MT5
2. Monitoriza simboluri cu priorități optimizate (GBPUSD first!)
3. Afișa heartbeat la fiecare 5 minute
4. Loga motivul pentru fiecare semnal respins (în mode DEBUG)
5. Aplica setări stricte pentru simbolurile cu performanță slabă
6. Risca 0.5% per trade
7. Limita la max 4 trades/zi