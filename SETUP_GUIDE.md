# Trading Bot - Setup Guide

## Ghid Complet pentru Rularea Botului pe Conturi Demo/Live

---

## 1. CERINTE PRELIMINARE

### Software Necesar

1. **Python 3.10+** - [Download Python](https://www.python.org/downloads/)
2. **MetaTrader 5** - [Download MT5](https://www.metatrader5.com/en/download)
3. **Git** (optional) - pentru versionare

### Cont de Trading

Pentru a rula botul, ai nevoie de un cont la un broker care suporta MT5:

**Brokeri Recomandati pentru Demo:**
- **MetaQuotes Demo** - cont demo gratuit direct din MT5
- **IC Markets** - spre zero, latenta scazuta
- **Pepperstone** - spread-uri mici
- **FTMO** - pentru prop trading challenges

---

## 2. INSTALARE

### Pasul 1: Cloneaza/Descarca proiectul

```bash
git clone <repository_url>
cd Trading_Bot
```

### Pasul 2: Creaza un mediu virtual

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### Pasul 3: Instaleaza dependentele

```bash
pip install -r requirements.txt
```

Sau manual:
```bash
pip install MetaTrader5 pandas numpy pytz pydantic pydantic-settings yfinance beautifulsoup4 requests
```

### Pasul 4: Configureaza variabilele de mediu

```bash
# Copiaza fisierul exemplu
copy .env.example .env

# Editeaza .env cu datele tale
notepad .env
```

---

## 3. CONFIGURARE METATRADER 5

### Pasul 1: Deschide un cont demo

1. Deschide MetaTrader 5
2. File -> Open an Account
3. Selecteaza "MetaQuotes-Demo" (sau alt broker)
4. Completeaza formularul pentru cont demo
5. Noteaza: **Account Number**, **Password**, **Server**

### Pasul 2: Configureaza .env

Editeaza fisierul `.env`:

```env
# Datele contului tau MT5
MT5_LOGIN=9042483
MT5_PASSWORD=J*8iFlAi
MT5_SERVER=Teletrade-Sharp ECN

# Path la MT5 (verifica sa fie corect)
MT5_PATH=C:\Program Files\MetaTrader 5\terminal64.exe

# Setari trading
DEFAULT_SYMBOL=NVDA
DEFAULT_TIMEFRAME=M5
TRADING_MODE=paper

# Risk management
MAX_RISK_PER_TRADE=1.0
MAX_DAILY_LOSS=3.0
```

### Pasul 3: Activeaza Algo Trading in MT5

1. Deschide MT5
2. Tools -> Options -> Expert Advisors
3. Bifeaza: "Allow algorithmic trading"
4. Bifeaza: "Allow DLL imports"
5. Click OK

---

## 4. MODURI DE RULARE

### A. Paper Trading (Simulare - Fara ordine reale)

**Recomandat pentru testare initiala!**

```bash
python run_live.py --mode paper --symbol NVDA
```

- Nu plaseaza ordine reale
- Afiseaza semnalele pe care le-ar genera
- Perfect pentru a verifica ca totul functioneaza

### B. Demo Trading (Ordine pe cont demo)

```bash
python run_live.py --mode demo --symbol NVDA
```

- Plaseaza ordine reale pe contul demo
- Bani virtuali, dar executie reala
- Testeaza strategia in conditii de piata reale

### C. Live Trading (ATENTIE - Bani reali!)

```bash
python run_live.py --mode live --symbol NVDA
```

- **ATENTIE**: Plaseaza ordine cu bani reali!
- Necesita confirmare explicita (trebuie sa tastezi "YES")
- Asigura-te ca ai testat pe demo inainte

---

## 5. BACKTESTING

### Backtesting cu Date Reale (Recomandat)

```bash
# Test pe un singur simbol
python run_backtest_real.py --symbol NVDA --days 60

# Test pe toate simbolurile (NVDA, AMD, TSLA)
python run_backtest_real.py --all --days 60

# Fara filtru de stiri
python run_backtest_real.py --all --no-news-filter
```

### Backtesting cu Date Generate

```bash
python run_backtest.py
```

---

## 6. MULTI-ACCOUNT TRADING

Pentru a rula pe mai multe conturi simultan:

### Pasul 1: Configureaza conturile

```bash
python multi_account/account_manager.py
```

Aceasta creeaza `config/accounts.json` cu o configuratie exemplu.

### Pasul 2: Editeaza accounts.json

```json
{
  "accounts": [
    {
      "account_id": "demo_001",
      "name": "Demo Account 1",
      "provider": "mt5",
      "login": "12345678",
      "password": "your_password",
      "server": "MetaQuotes-Demo",
      "initial_balance": 10000,
      "max_daily_drawdown": 4.0,
      "symbols": ["NVDA", "AMD"]
    }
  ]
}
```

### Pasul 3: Ruleaza multi-account

```bash
python run_live.py --multi
```

---

## 7. FILTRUL PENTRU STIRI

Botul include un filtru automat care:
- Evita zilele cu stiri cu impact ridicat (NFP, FOMC, CPI)
- Marcate cu rosu in Forex Factory

### Verificare manuala stiri

```bash
python -c "from utils.news_filter import create_news_filter; f = create_news_filter(); f.print_calendar(30)"
```

### Dezactivare filtru stiri

```bash
python run_live.py --mode paper --no-news-filter
```

---

## 8. PARAMETRI STRATEGIEI

### Parametri Cheie (in .env sau direct)

| Parametru | Valoare Implicita | Descriere |
|-----------|-------------------|-----------|
| ORB_RANGE_MINUTES | 5 | Durata opening range |
| RISK_REWARD_RATIO | 2.5 | Target R:R |
| MAX_RISK_PER_TRADE | 1.0% | Risk per trade |
| MAX_DAILY_LOSS | 3.0% | Max pierdere zilnica |
| USE_VWAP_FILTER | true | Filtru VWAP |
| USE_EMA_FILTER | true | Filtru EMA 20/50 |
| USE_ATR_FILTER | true | Filtru volatilitate |

---

## 9. SCHEDULE ZILNIC RECOMANDAT

1. **08:00-09:00** - Verifica calendarul economic (news filter)
2. **09:25** - Porneste botul in modul paper
3. **09:30** - Deschiderea pietei - botul calculeaza Opening Range
4. **09:35-11:30** - Fereastra de trading principala
5. **14:00-15:30** - Fereastra secundara de trading
6. **15:50** - Botul inchide pozitiile inainte de close
7. **16:00** - Inchiderea pietei

---

## 10. TROUBLESHOOTING

### Eroare: "MT5 initialization failed"

1. Verifica ca MT5 este instalat si deschis
2. Verifica path-ul in .env
3. Asigura-te ca ai bicat "Allow DLL imports" in MT5

### Eroare: "Login failed"

1. Verifica credentialele in .env
2. Verifica ca serverul este corect
3. Testeaza login-ul manual in MT5

### Eroare: "Symbol not found"

1. Verifica ca simbolul exista la broker
2. Pentru actiuni US, verifica daca broker-ul le ofera
3. Incearca cu indici: US500, US30, USTEC

### Nu primesc semnale

1. Verifica ca esti in orele de trading (09:30-16:00 NY)
2. Verifica ca nu este o zi cu stiri (news filter activ)
3. Ruleaza cu `--log-level DEBUG` pentru mai multe detalii

---

## 11. SFATURI PENTRU PROP TRADING

### Reguli FTMO/MyForexFunds/TopStep

1. **Max Daily Drawdown**: 5% (configureaza MAX_DAILY_LOSS=4%)
2. **Max Account Drawdown**: 10%
3. **Profit Target**: 10% (phase 1), 5% (phase 2)
4. **Min Trading Days**: 4-10 zile

### Configurare pentru Prop Trading

```env
MAX_RISK_PER_TRADE=0.5
MAX_DAILY_LOSS=4.0
MAX_POSITIONS=1
```

### Reguli de Aur

1. **NICIODATA** nu tranzactiona in zilele cu NFP, FOMC, CPI
2. Nu supratrada - max 1-2 trade-uri pe zi
3. Opreste-te dupa 2 pierderi consecutive
4. Ia profit partial la 1:1 R:R

---

## 12. COMENZI UTILE

```bash
# Backtesting
python run_backtest_real.py --all --days 60

# Paper trading
python run_live.py --mode paper --symbol NVDA

# Demo trading
python run_live.py --mode demo --symbol NVDA

# Verifica stiri
python -c "from utils.news_filter import create_news_filter; f = create_news_filter(); f.print_calendar(14)"

# Multi-account setup
python multi_account/account_manager.py

# Parametru optimizare
python run_backtest.py --optimize
```

---

## 13. STRUCTURA PROIECTULUI

```
Trading_Bot/
├── brokers/           # Conexiuni la brokeri (MT5, NinjaTrader)
├── strategies/        # Strategii de trading (ORB, ORB+VWAP)
├── backtesting/       # Engine de backtesting
├── utils/             # Utilitare (risk manager, news filter, indicators)
├── multi_account/     # Management multi-cont
├── config/            # Configurari
├── data/              # Date si rapoarte
├── logs/              # Log-uri
├── run_live.py        # Script trading live/demo
├── run_backtest.py    # Script backtesting simplu
├── run_backtest_real.py  # Script backtesting cu date reale
└── main.py            # Entry point principal
```

---

## CONTACT SI SUPORT

Pentru intrebari sau probleme:
1. Verifica log-urile in `logs/`
2. Ruleaza cu `--log-level DEBUG`
3. Testeaza pe cont demo inainte de live

**DISCLAIMER**: Acest bot este pentru uz educational. Trading-ul comporta riscuri. Nu investi mai mult decat iti permiti sa pierzi.
