# Claude AI - Starea Proiectului BOT_TRADING

Acest fișier oferă un rezumat al stării curente a proiectului și a modificărilor implementate în cadrul sesiunilor cu Gemini CLI, pentru a facilita colaborarea.

---

## INSTRUCȚIUNI PENTRU CLAUDE AI

### La Începutul Fiecărei Conversații:
1.  **CITEȘTE ACEST FIȘIER** - conține starea curentă a proiectului.
2.  Verifică ce faze sunt completate și ce trebuie făcut.
3.  Continuă de unde ai rămas, înțelegând contextul modificărilor recente.
4.  După modificări majore, actualizează acest fișier.

---

## STATUS CURENT AL PROIECTULUI

### Ultima Actualizare: 2025-12-21 (v2 - Claude Opus)

### Strategie Activă: SMC V3 (OPTIMIZAT PER-SIMBOL)

**Fișiere principale:**
-   `strategies/smc_strategy_v3.py` - Logica strategiei SMC V3 (implementată).
-   `scripts/run_smc_v3.py` - Script pentru rulare live/demo/paper (orchestrator, management multi-simbol).
-   `backtests/backtest_multi_instrument.py` - Backtest multi-simbol (modificat pentru M1/M5 yfinance data).
-   `backtesting/data_loader.py` - Încărcare date (modificat pentru 1m/5m yfinance data).

---

## REZUMAT MODIFICĂRI ȘI IMPLEMENTĂRI RECENTE (CU Gemini CLI)

Am finalizat implementarea logicii core a strategiei SMC V3 și am corectat numeroase probleme de configurare și erori de runtime.

### Modificări Cheie:
*   **`smc_strategy_v3.py` (Strategia SMC V3):**
    *   Am implementat calculul indicatorilor **ATR, EMA, ADX** cu robustețe sporită.
    *   Am adăugat logica pentru **Swing Points, BOS/ChoCH**, **Order Blocks, Fair Value Gaps** și **Niveluri de Lichiditate**.
    *   Am construit fluxul principal `on_candle` pentru **generarea semnalelor** și `should_exit` pentru **gestionarea ieșirilor bazate pe timp**.
    *   Am asigurat **ajustarea Stop Loss-ului pentru spread** și am rezolvat `SyntaxError`-uri și `AttributeError`-uri legate de `SignalType`.
    *   Logica `_find_structure_breaks` a fost refactorizată pentru o detecție mai precisă a BOS/ChoCH.
*   **`run_smc_v3.py` (Orchestrator):**
    *   Am corectat configurațiile `min_rr` pentru GBPUSD și `adx_trending` pentru US30.
    *   Am verificat integrarea cu `RiskManager`, `MT5Broker` și `NewsFilter`.
*   **`data_loader.py` (Încărcare Date):**
    *   A fost îmbunătățită gestionarea datelor de 1 minut de la Yahoo Finance, cu fallback robust la datele de 5 minute pentru perioade lungi și generarea de date de 1 minut resamplerizate.
*   **`backtest_multi_instrument.py` (Engine Backtest):**
    *   A fost adaptat pentru a rula pe date de 1 minut (sau resamplerizate din 5 minute).
    *   Am rezolvat probleme de compatibilitate și erori de runtime (`AttributeError`, `SyntaxError`) pentru a permite rularea backtest-ului.

### Stare Funcțională:
Botul este acum **aproape funcțional** și pregătit pentru testare extensivă pe conturi demo. Logica core a strategiei a fost implementată, iar problemele majore de integrare și sintaxă au fost rezolvate.

---

## RULARE BOT (DEMO CU LOGURI DETALIATE)

```bash
python Bot_Trading/scripts/run_smc_v3.py --mode demo --log-level DEBUG --risk 0.5
```

---

## NEXT STEPS / TODO

### Prioritate 1: Testare Demo și Monitorizare
-   [ ] Rulați botul în modul `demo` cu `log-level DEBUG` pe toate simbolurile.
-   [ ] Monitorizați logurile și comportamentul botului.

### Prioritate 2: Optimizarea Strategiei și Extinderea Funcționalităților
-   [ ] **Îmbunătățirea Strategiei de Ieșire:** Implementați trailing stop și/sau TP-uri parțiale avansate.
-   [ ] **Rafinarea Logicii ADX:** Investigați și rezolvați definitiv avertismentele `RuntimeWarning` pentru o stabilitate numerică completă.
-   [ ] **Optimizarea Parametrilor:** Ajustați parametrii strategiei pe baza performanței demo.

### Prioritate 3: Adresarea Lentoarei Backtesting-ului
-   [ ] Investigați motivele pentru care backtesting-ul durează atât de mult (posibile cauze: volum mare de date de 1 minut, procesare intensivă a indicatorilor, ineficiențe în bucla de backtest). Odată ce botul este validat, putem aborda această problemă.

### Prioritate 4: Refactorizare Proiect
-   [ ] Propuneți și implementați o structură mai organizată a pachetelor/fișierelor.

---

## MODIFICARI SESIUNE CLAUDE OPUS (2025-12-21)

### Agenti Specializati Creati
S-au creat 6 agenti slash commands in `.claude/commands/`:
- `/backtest` - Rulare si analiza backtesting
- `/validate` - Validare modificari conform plan
- `/find` - Cautare fisiere si cod rapid
- `/debug` - Verificare erori si bug-uri
- `/terminal` - Analiza output terminal
- `/coordinator` - Coordonare generala

### Bug-uri Critice Reparate
1. **`SignalType.CLOSE_POSITION`** - Adaugat in `base_strategy.py`
2. **Import lipsa** - Adaugat `from strategies.base_strategy import SignalType` in `backtest_multi_instrument.py`
3. **`on_position_closed`** - Implementat in `smc_strategy_v3.py`
4. **`get_status`** - Imbunatatit in `smc_strategy_v3.py` cu date SMC-specific

### LOCB Strategy - Optimizari
1. **Sesiuni corecte**: London 8-11 UTC, NY 14-17 UTC (era 2-4 si 8-10!)
2. **Displacement**: Crescut de la 4 pips la 8 pips (reduce noise)
3. **OC Range**: Crescut de la 3-12 la 5-15 pips
4. **SL minim**: Adaugat 8 pips minim (era doar 2 pips!)
5. **R:R**: Redus de la 2.5 la 2.0 (mai realist)

### Probleme Identificate (de rezolvat)
1. Logica BOS/CHoCH poate fi inversata in `_find_structure_breaks`
2. Position in backtest e dict, nu Position object - poate cauza probleme
3. FVG detection itereaza backwards - confuz dar pare corect
