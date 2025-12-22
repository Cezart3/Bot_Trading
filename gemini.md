# Gemini CLI - Starea Proiectului BOT_TRADING

Acest fișier oferă un rezumat al stării curente a proiectului și a modificărilor implementate în cadrul sesiunilor cu Gemini CLI.

---

## STATUS CURENT AL PROIECTULUI

### Ultima Actualizare: 2025-12-22
### Strategie Activă: ORB (Asian Range Breakout - "Claude" Version)
**Status:** In curs de backtesting și optimizare pentru Prop Firm.

**Schimbare Majoră:**
-   **SMC / LOCB V3:** DEPRECATED / PAUSED. Strategiile anterioare (SMC, LOCB Simple/V3) nu au oferit rezultatele dorite.
-   **FOCUS NOU:** Strategia **ORB (Opening Range Breakout)** pe sesiunea Asiatică.

---

## DETALII STRATEGIE (ORB - Asian Session)

**Logica de bază:**
1.  **Asian Range:** Se definește intervalul 00:00 - 08:00 UTC (02:00 - 10:00 România). Se marchează High și Low.
2.  **London Breakout:** Între 08:00 - 11:00 UTC (10:00 - 13:00 România), se caută spargerea range-ului.
3.  **Filtre:**
    *   **ADX (14):** Trebuie să fie > 20 (indică prezența unui trend).
    *   **Range Height:** Distanța High-Low trebuie să fie între 10 și 50 pips (evităm range-uri prea mici = noise, sau prea mari = mișcare epuizată).
4.  **Intrare:** Breakout confirmat (închidere lumânare M15 sau M5 în afara range-ului).
5.  **Risk Management:**
    *   **Risc:** 0.5% per trade.
    *   **SL:** Capătul opus al range-ului (sau calculat dinamic).
    *   **TP:** R:R 2.0.

---

## OBIECTIVE PROP FIRM

Parametrii stricți pentru validare:
-   **Risc per tranzacție:** 0.5%
-   **Max Daily Drawdown:** 3% (Hard limit: 4%)
-   **Max Total Drawdown:** 8% (Hard limit: 8%)
-   **Parități Target:** EURUSD, GBPUSD, EURGBP, GBPJPY, USDJPY, XAUUSD.

---

## FIȘIERE IMPORTANTE

-   `strategies/orb_forex_strategy.py` (sau `orb_strategy.py`): Implementarea logicii Asian Range ORB.
-   `backtests/run_historical_backtest.py`: Scriptul de simulare cu reguli Prop Firm.
-   `data/historical/`: Datele CSV pentru backtesting.

---

## TODO LIST

1.  [x] Actualizare `gemini.md` cu noua direcție.
2.  [ ] Verificare/Implementare logica ORB "Claude" în cod (`orb_forex_strategy.py`).
3.  [ ] Rulare Backtest Istoric complet pe toate paritățile disponibile.
4.  [ ] Analiza rezultatelor (Win Rate, Drawdown, Profit).
5.  [ ] Activare Live Trading (dacă rezultatele sunt pozitive).