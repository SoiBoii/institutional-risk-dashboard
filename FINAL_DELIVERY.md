# Institutional Cyberpunk Portfolio Risk Dashboard — Final Delivery

## Project Summary
A production-ready, full-stack quantitative finance SaaS platform combining advanced ML models, live market data, and a stunning Cyberpunk HUD interface.

---

## 1. Core Files

| File | Description |
|---|---|
| `app.py` | Flask backend — quantitative engine, ML inference, SaaS API |
| `index.html` | Single-page frontend — Cyberpunk HUD, Chart.js, auth flows |
| `models.py` | SQLAlchemy database models (User, Portfolio, Transaction, Watchlist, AccountHistory) |
| `requirements.txt` | Python dependency manifest |

---

## 2. `requirements.txt`
```txt
flask
flask-cors
flask-sqlalchemy
flask-login
werkzeug
yfinance
pandas
numpy
scipy
textblob
scikit-learn
statsmodels
```

---

## 3. Key API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Serves `index.html` SPA |
| `/register` | POST | Creates a new user account |
| `/login` | POST | Authenticates user, starts session |
| `/logout` | POST | Terminates session |
| `/user_info` | GET | Returns current user info & account value |
| `/analyze` | POST | Full quantitative analysis engine |
| `/api/auth/status` | GET | Checks auth status for session restore |
| `/api/portfolio/save` | POST | Persists current portfolio to `_DEFAULT_SESSION_` |
| `/api/portfolio/load` | GET | Loads saved portfolio session |
| `/api/simulate_trade` | POST | Simulates a BUY/SELL, recalculates weights |
| `/api/ml/anomaly` | POST | IsolationForest anomaly detection on asset |
| `/api/ml/forecast` | POST | ARIMA 30-day price forecast with 95% CI |
| `/save_portfolio` | POST | Saves a named portfolio layout |
| `/load_portfolios` | GET | Lists all saved layouts |
| `/watchlist` | GET/POST/DELETE | Manages user watchlist |
| `/leaderboard` | GET | Returns global user leaderboard |

---

## 4. ML Features

### Market Anomaly Detection (`/api/ml/anomaly`)
- Uses `scikit-learn` IsolationForest trained on 2 years of daily returns + volume.
- Contamination parameter: `5%`.
- Returns anomalous dates and prices for frontend chart overlay.

### Neural Projection Engine (`/api/ml/forecast`)
- Uses `statsmodels` ARIMA(5,1,0) trained on 2 years of closing prices.
- Returns a 30-day forward price path, 95% confidence intervals, and an $R^2$ accuracy score.

---

## 5. Persistent Session Architecture

- On page load, `initApp()` pings `/api/auth/status`.
- If authenticated, it loads the `_DEFAULT_SESSION_` portfolio from the DB, populates all input fields, and auto-fires the analysis engine.
- Every successful analysis auto-saves the configuration back to the DB (non-blocking background fetch).
- Guest portfolios are captured before login and immediately persisted after successful auth.

---

## 6. How to Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Start the server
python3 app.py

# 3. Access the dashboard
# Open http://127.0.0.1:5050 in your browser
```
