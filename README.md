# Advanced Institutional Portfolio Risk & Analytics Platform

This project is a production-ready, full-stack quantitative finance application. It provides an institutional-grade portfolio risk dashboard, combining a robust Python backend with a sleek, interactive "Cyberpunk" HTML/JS frontend.

The platform is designed to ingest real-time market data, perform advanced statistical mathematics (Optimization, Monte Carlo, Linear Regression), and deliver actionable insights including sentiment analysis and trade generation.

---

## 🛠 System Architecture

The application is split into two primary components:
1. **Backend (`app.py`)**: A Flask REST API that acts as the quantitative engine. It fetches data via Yahoo Finance, processes it using NumPy/Pandas/SciPy, and serves JSON payloads.
2. **Frontend (`index.html`)**: A single-page application built with Vanilla JavaScript, Tailwind CSS, and Chart.js, designed as a highly interactive "Heads-Up Display" (HUD).

---

## 🧮 Backend Mathematics & Logic (`app.py`)

The backend performs several complex quantitative computations. Here is a detailed breakdown of the math and logic:

### 1. Expected Return & Volatility
Daily percentage returns are calculated from Adjusted Close prices. 
*   **Annualized Return**: $\mu_{annual} = \mu_{daily} \times 252$
*   **Portfolio Variance**: $\sigma^2_p = w^T \cdot \Sigma \cdot w$ (where $w$ is the weight vector and $\Sigma$ is the covariance matrix)
*   **Annualized Volatility**: $\sigma_{annual} = \sqrt{\sigma^2_p} \times \sqrt{252}$

### 2. Portfolio Optimization (SciPy SLSQP)
We use SciPy's Sequential Least Squares Programming (SLSQP) optimizer to find two target portfolios:
*   **Maximum Sharpe Portfolio**: Maximizes the risk-adjusted return. The objective function minimizes the negative Sharpe Ratio: $-(R_p - R_f) / \sigma_p$.
*   **Minimum Volatility Portfolio**: Minimizes the absolute portfolio variance $w^T \cdot \Sigma \cdot w$.
Constraints: All weights must sum to 1 ($\sum w_i = 1$) and no short selling ($0 \le w_i \le 1$).

### 3. Conditional Value at Risk (CVaR)
Also known as Expected Shortfall, CVaR evaluates extreme downside risk. 
*   First, we calculate the 95% Value at Risk (VaR), which is the 5th percentile of historical daily returns.
*   **CVaR**: The mathematical average of all returns that fall *below* the VaR threshold. It answers: "If the worst-case scenario happens, how much are we expected to lose?"

### 4. Geometric Brownian Motion (GBM) Monte Carlo
To project wealth 10 years into the future, we simulate thousands of random paths using the GBM stochastic differential equation:
$dS = \mu S dt + \sigma S dW$
*   **Drift**: $(\mu - \frac{1}{2}\sigma^2)dt$
*   **Shock**: $\sigma \sqrt{dt} \times Z$ (where Z is a standard normal random variable)
The paths are aggregated to determine the 10th (bear), 50th (base), and 90th (bull) percentile outcomes over time.

### 5. Factor Exposure (Multiple Linear Regression)
To understand what macroeconomic factors drive the portfolio, we run a Least-Squares Linear Regression of the portfolio's daily returns against three proxy indices:
*   `^GSPC` (S&P 500) -> Market Beta
*   `^RUT` (Russell 2000) -> Size Factor (Small-Cap vs Large-Cap)
*   `VTV` (Vanguard Value ETF) -> Value Factor (Value vs Growth)
The resulting coefficients (betas) are exposed to the frontend to build the Factor Overlay Radar Chart.

### 6. Natural Language Processing (NLP) Sentiment
Using `TextBlob`, the backend pulls the latest news headlines for the targeted assets from Yahoo Finance.
*   **Polarity Score**: Each headline is scored from `-1.0` (Negative) to `1.0` (Positive).
*   Scores $> 0.1$ are tagged `BULL`, scores $< -0.1$ are tagged `BEAR`, and everything else is `NEUTRAL`.

### 7. Trade Rebalancing Engine
Accepts a `Total Capital` input and calculates the exact dollar and share flows required to transition from the User's inputted weights to the Optimal (Max Sharpe) weights.
*   **Dollar Flow**: $(Capital \times Optimal\_Weight) - (Capital \times Current\_Weight)$
*   **Shares to Trade**: $Dollar\_Flow / Latest\_Price$

---

## 🖥 Frontend Implementation (`index.html`)

The UI is a state-of-the-art Cyberpunk HUD built without heavy frameworks (like React or Angular) to guarantee maximum performance and minimal overhead.

### 1. Institutional Cyberpunk Aesthetic
*   **Styling**: Uses Tailwind CSS via CDN. Colors rely on deep obsidian backgrounds (`#09090b`), translucent glass panels (`bg-white/5 backdrop-blur-md`), and glowing neon accents (Cyan, Electric Purple, Toxic Green, Neon Red).
*   **Typography**: Uses Google Fonts (`Inter` and `Space Mono`) to create a command-line terminal feel.

### 2. Universal Deep-Dive Modals
Every KPI card and Chart container has `cursor-pointer` and hover effects. When clicked, a JavaScript router `openDeepDive(type, title, subtitle)` triggers a massive overlay modal.
*   This modal destroys the previous `<canvas>` and injects a new `Chart.js` instance specifically configured for a full-screen, high-resolution deep-dive.

### 3. Execution Terminal & Live Comm-Link
*   **Execution Terminal**: Renders the backend's Rebalancing calculations as a monospaced table. It conditionally applies text colors (`text-cyber-green` for BUY, `text-cyber-red` for SELL) to give the user immediate visual cues.
*   **Live Comm-Link**: Maps the NLP sentiment JSON array into a responsive CSS grid of "News Cards", featuring pulsing CSS animations to emulate a live satellite feed.

### 4. Chart.js Visualizations
Extensively utilizes `Chart.js` for complex graphing:
*   **Scatter Plot**: Used for the Efficient Frontier, mapping 1,000+ random portfolios as points with varying colors based on their Sharpe ratio.
*   **Line Charts**: Used for Cumulative Returns, Rolling Correlation, and the Monte Carlo wealth projections (utilizing area fills for percentiles).
*   **Radar Chart**: Used to visualize the multidimensional Factor Exposure Betas.

---

## 🚀 How to Run

### Requirements
*   Python 3.8+
*   `pip install flask flask-cors yfinance numpy pandas scipy textblob`

### Execution
1.  **Start the Backend**:
    Open a terminal and run:
    ```bash
    python3 app.py
    ```
    *The API will start on `http://127.0.0.1:5000`.*

2.  **Start the Frontend**:
    Open a second terminal in the project directory and run:
    ```bash
    python3 -m http.server 8000
    ```
    *Navigate your browser to `http://localhost:8000/index.html`.*

3.  **Analyze**:
    Input your comma-separated tickers (e.g., `AAPL, MSFT, GOOG, NVDA`), their weights (e.g., `0.25, 0.25, 0.25, 0.25`), your total capital, and click **EXECUTE**.
