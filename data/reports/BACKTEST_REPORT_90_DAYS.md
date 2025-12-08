# RAPORT BACKTEST ORB STRATEGY
## Perioada: 13 Octombrie - 5 Decembrie 2025 (39 zile de trading)

---

## SUMAR EXECUTIV

### Rezultate Agregate (NVDA + AMD + TSLA)

| Metric | Valoare |
|--------|---------|
| **Total Trade-uri** | 109 |
| **Trade-uri Castigatoare** | 37 (33.9%) |
| **Trade-uri Pierzatoare** | 72 |
| **Profit Total** | $661.63 |
| **Comisioane Totale** | $218.00 |
| **Profit Net** | $443.63 |
| **Simboluri Profitabile** | 2/3 (AMD, TSLA) |

---

## REZULTATE PE SIMBOL

### 1. NVDA (NVIDIA)

| Metric | Valoare |
|--------|---------|
| Balance Initial | $25,000 |
| Balance Final | $22,580 |
| **Profit/Pierdere** | **-$2,345.79 (-9.7%)** |
| Total Trade-uri | 37 |
| Win Rate | 24.3% |
| Profit Factor | 0.64 |
| Max Drawdown | 10.5% |
| Average Win | $461.18 |
| Average Loss | $232.02 |

**Analiza:** NVDA a avut performanta slaba in aceasta perioada. Volatilitatea ridicata si trendurile puternice au dus la multe stop-loss-uri. Recomandat: evitarea NVDA in perioadele de volatilitate extrema (earnings, FOMC).

---

### 2. AMD (Advanced Micro Devices)

| Metric | Valoare |
|--------|---------|
| Balance Initial | $25,000 |
| Balance Final | $27,468 |
| **Profit/Pierdere** | **+$2,539.95 (+9.9%)** |
| Total Trade-uri | 36 |
| Win Rate | 38.9% |
| Profit Factor | 1.46 |
| Max Drawdown | 7.4% |
| Average Win | $576.00 |
| Average Loss | $251.09 |

**Analiza:** AMD a fost cel mai profitabil simbol. Profit Factor de 1.46 indica o strategie viabila. Raportul castig/pierdere de 2.29:1 compenseaza win rate-ul de sub 40%.

---

### 3. TSLA (Tesla)

| Metric | Valoare |
|--------|---------|
| Balance Initial | $25,000 |
| Balance Final | $25,395 |
| **Profit/Pierdere** | **+$467.47 (+1.6%)** |
| Total Trade-uri | 36 |
| Win Rate | 38.9% |
| Profit Factor | 1.06 |
| Max Drawdown | 8.8% |
| Average Win | $575.74 |
| Average Loss | $345.13 |

**Analiza:** TSLA a fost marginal profitabil. Profit Factor de 1.06 sugereaza ca strategia functioneaza, dar nu optim pentru acest simbol.

---

## DISTRIBUTIA REZULTATELOR

### Pe Tip de Iesire

| Tip Iesire | Trade-uri | PnL Total |
|------------|-----------|-----------|
| Take Profit (TP) | 24 | +$14,920.29 |
| Stop Loss (SL) | 70 | -$19,554.90 |
| Session End | 15 | +$5,296.24 |

**Observatii:**
- 22% din trade-uri au atins TP (target 2.5:1 R:R)
- 64% din trade-uri au atins SL
- 14% au fost inchise la finalul sesiunii (unele profitabile)

---

## ANALIZA TEMPORALA

### Cele Mai Bune Zile de Trading

1. **11 Nov** - Multiple trade-uri profitabile pe AMD si NVDA
2. **13 Nov** - TSLA TP +$958, NVDA TP +$763
3. **24 Nov** - TSLA TP +$1,138, AMD TP +$743

### Cele Mai Slabe Zile de Trading

1. **5 Nov** - NVDA SL -$254
2. **6 Nov** - NVDA SL -$288 (dupa profit pe SHORT)
3. **18 Nov** - Multiple SL pe toate simbolurile

---

## METRICI DE RISC

| Metric | NVDA | AMD | TSLA | Aggregate |
|--------|------|-----|------|-----------|
| Max Drawdown % | 10.5% | 7.4% | 8.8% | ~9% |
| Sharpe Ratio | -0.31 | 0.27 | 0.05 | ~0.00 |
| Consecutive Losses | 8 | 4 | 5 | - |
| Recovery Factor | 0.89 | 1.33 | 0.19 | - |

---

## RECOMANDARI PENTRU IMBUNATATIRE

### 1. Selectia Simbolurilor
- **Recomandare**: Focus pe AMD ca simbol principal
- NVDA: reduce dimensiunea pozitiei sau evita in perioadele volatile
- TSLA: tranzactioneaza doar cand volatilitatea este normala

### 2. Filtre Aditionale
- **News Filter**: ACTIV (fara trade-uri in zilele NFP, FOMC, CPI)
- **ATR Filter**: Evita zilele cu ATR > 3x media
- **Time Filter**: Focus pe 09:35-11:30 (morning momentum)

### 3. Ajustari Risk Management
```
Current:
- Risk per trade: 1%
- R:R Ratio: 2.5:1
- Max Daily Loss: 3%

Recomandat pentru Prop Trading:
- Risk per trade: 0.5% (mai conservator)
- R:R Ratio: 2.5:1 (mentine)
- Max Daily Loss: 2% (opreste dupa 4 pierderi consecutive)
```

### 4. Reguli de Oprire
- Stop dupa 2 pierderi consecutive
- Stop daca drawdown-ul zilnic > 2%
- Nu tranzactiona luni si vineri (volatilitate mai mare)

---

## PROIECTII VIITOARE

### Scenariu Conservator (AMD only, 0.5% risk)
- Trade-uri estimate/luna: ~15
- Win Rate asteptat: 35-40%
- Return lunar estimat: 3-5%
- Drawdown maxim asteptat: 8%

### Scenariu Moderat (AMD + TSLA, 0.75% risk)
- Trade-uri estimate/luna: ~25
- Win Rate asteptat: 35%
- Return lunar estimat: 2-4%
- Drawdown maxim asteptat: 10%

---

## CONCLUZIE

Strategia ORB cu filtre VWAP, EMA si ATR arata potentiaal pentru trading profitabil pe AMD. NVDA necesita ajustari suplimentare sau evitare in perioadele foarte volatile.

**VERDICT**: Strategia este **VIABILA** dar necesita:
1. Focus pe simboluri mai stabile (AMD > TSLA > NVDA)
2. Risk management mai strict pentru prop trading
3. Evitarea zilelor cu stiri importante (news filter ACTIV)

---

## PASI URMATORI

1. [ ] Ruleaza 2 saptamani pe cont DEMO cu parametrii recomandati
2. [ ] Monitorizeaza zilnic rezultatele
3. [ ] Ajusteaza parametrii daca drawdown > 5%
4. [ ] Dupa validare pe demo, treci pe cont real cu 50% din pozitia normala
5. [ ] Creste gradual la pozitia completa dupa 30 de zile profitabile

---

*Raport generat: 8 Decembrie 2025*
*Perioada analizata: 13 Oct - 5 Dec 2025 (39 trading days)*
*Strategie: ORB + VWAP + EMA + ATR Filter*
