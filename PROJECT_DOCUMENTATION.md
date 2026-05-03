# Institutional Cyberpunk Portfolio Risk Dashboard & SaaS Platform

## 1. Project Overview
The Institutional Cyberpunk Portfolio Risk Dashboard is a full-stack, production-ready SaaS platform built for modern quantitative analysts, day-traders, and financial enthusiasts. It merges highly advanced financial mathematics (Modern Portfolio Theory, Monte Carlo Simulations, NLP Sentiment Analysis) with a strikingly beautiful, immersive "Cyberpunk HUD" aesthetic. 

Far more than a simple static calculator, this application features a persistent backend architecture that tracks user sessions, manages dynamic simulated portfolios (paper trading), and compares user performance across a global leaderboard.

---

## 2. Technology Stack
* **Frontend UI/UX:** Vanilla JavaScript, HTML5, Tailwind CSS (via CDN for glassmorphic styling and neon aesthetics), Chart.js (for data visualization), html2pdf.js (for reporting).
* **Backend API:** Python 3, Flask.
* **Database & Auth:** SQLite, Flask-SQLAlchemy, Flask-Login, Werkzeug Security (Password Hashing).
* **Data Ingestion & Quant Libraries:** `yfinance` (Live market data), `pandas`, `numpy`, `TextBlob` (NLP).
* **Machine Learning & Forecasting:** `scikit-learn` (IsolationForest), `statsmodels` (ARIMA).

---

## 3. Core Quantitative Engine
The backend serves as a high-powered calculation engine, completely abstracting the heavy lifting from the browser.
* **Efficient Frontier & MPT:** Calculates covariance matrices to plot the optimal risk/reward frontier for any given basket of assets.
* **Monte Carlo Simulations:** Runs 1,000+ stochastic simulations to project potential wealth paths over 10 years, plotting the 10th, 50th, and 90th percentile outcomes.
* **Asset Telemetry & Anomaly Detection:** Tracks standard deviation, trailing sparklines, and uses an Isolation Forest ML algorithm to detect extreme volume/price spikes, placing neon warning overlays on charts.
* **Neural Projection Engine:** Uses an ARIMA(5,1,0) autoregressive time-series model to calculate a forward 30-day predicted price trajectory with expanding 95% confidence intervals and an $R^2$ model confidence score.
* **NLP News Sentiment:** Fetches recent headlines for portfolio assets and runs Natural Language Processing (via TextBlob) to score them as Bullish, Bearish, or Neutral.

---

## 4. SaaS Architecture & Authentication
The platform is fully session-aware. All user data is strictly barricaded behind encrypted HTTP-only cookies.
* **Multi-Tenant Environment:** Users can register accounts safely. Upon logging in, a visual "CRT Terminal Boot" animation transitions the DOM from a public state to their private Command Center.
* **Freemium Subscription Tiers:** Implements a strict tier system. "Standard" users are restricted to tracking a maximum of 3 assets per portfolio and 2 assets in their watchlist. Attempts to bypass this trigger a neon-red `ACCESS_DENIED` modal prompting an upgrade.
* **Multi-Portfolio Management:** Users can create, save, and seamlessly switch between multiple named layout configurations (e.g., "Tech Growth", "Dividend Safe").

---

## 5. Gamification & Institutional Features
### The Dynamic Trade Blotter (Paper Trading)
The trade blotter allows users to execute simulated live trades (Buy/Sell) across their assets. Upon entering an order, the system fetches the live market price, calculates the total dollar transaction, and instantly recomputes the global portfolio weights and capital distribution. It then logs the transaction visually and automatically retriggers a completely seamless re-analysis of the dashboard parameters in the background.

### The "Shadow Server" Leaderboard (Social Trading)
Trading in a vacuum is boring. The platform aggregates the Total Account Value of all users, normalizes it against initial capital, and ranks the top 10 traders dynamically on a global leaderboard modal. 

### Live Comm-Link Watchlist
A vertically stacked panel allowing users to track assets outside their main portfolio. The backend continuously evaluates trailing 24-hour performance. If an asset drops by more than **2%**, its respective UI card overrides standard CSS to trigger a highly visible, pulsing "neon red breathe" animation—alerting the user to a potential dip-buying opportunity.

### Historical Equity Curve
A personalized line-chart tracking the user's Total Account Net Worth. Every time the user logs in, the backend takes a snapshot of their total value, charting their actual trading success over weeks and months independently of the backtester.

### UI Personalization
Through the Settings modal, users can dynamically shift the entire application's CSS variables and Chart.js datasets from "Neon Cyan" to "Matrix Green" or "Synthwave Purple", which is saved directly to their user profile in the database.

### Institutional PDF Export
A one-click `[ PDF_TEARSHEET ]` function. It temporarily strips away the background, reorganizes the CSS Grid layout, and forces the browser to compile all 11 complex Chart.js canvases and data tables into a high-resolution, downloadable landscape PDF report for client presentation.

---

## 6. How to Run
1. Activate your virtual environment (if applicable).
2. Install dependencies: `pip install flask flask-cors flask-sqlalchemy flask-login werkzeug yfinance pandas numpy scipy textblob scikit-learn statsmodels`
3. Start the server: `python3 app.py`
4. Access the frontend dashboard: Navigate to `http://127.0.0.1:5050` in your browser. The single-page application is served natively through Flask.
