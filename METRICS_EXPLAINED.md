# Quantitative Metrics Guide

This document provides a detailed breakdown of every quantitative metric computed and displayed on the Institutional Risk & Analytics Dashboard. These metrics are essential in modern portfolio theory and institutional finance to balance expected rewards against downside risks.

---

### 1. Expected Annual Return
* **What it is:** The theoretical average amount your portfolio is expected to grow (or shrink) over a one-year period.
* **How it's calculated:** The backend calculates the daily percentage change (returns) of all your assets over your selected timeframe. It averages these daily returns and then annualizes them by multiplying by `252` (the standard number of trading days in a year). 
* **Why it matters:** It sets your baseline expectation for capital growth. However, looking at return *without* considering volatility is dangerous, which is why it is paired with the next metric.

### 2. Annualized Volatility (Risk)
* **What it is:** A measure of how wildly your portfolio's value swings up and down. Higher volatility means higher uncertainty and risk.
* **How it's calculated:** It uses the **Covariance Matrix** of your assets. The engine computes the standard deviation of your portfolio's daily returns (factoring in how the assets move in relation to one another) and scales it up to an annual figure by multiplying by `√252`.
* **Why it matters:** If two portfolios both have a 10% Expected Return, but Portfolio A has 5% volatility and Portfolio B has 25% volatility, Portfolio A is vastly superior because it provides a smoother, more predictable ride.

### 3. Sharpe Ratio
* **What it is:** The gold standard metric for **Risk-Adjusted Return**. It tells you how much excess return you are getting for every "unit" of risk you take on.
* **How it's calculated:** `(Expected Return - Risk-Free Rate) / Annualized Volatility`. *(Note: Your backend currently uses a 0% risk-free rate for simplicity, which is standard in many basic models).*
* **Why it matters:** A higher Sharpe Ratio is always better. 
  * `< 1.0` is sub-optimal.
  * `1.0` is good.
  * `> 2.0` is excellent (rare in long-only equity portfolios).
  * Your dashboard specifically calculates the "Max Sharpe" weights to mathematically find the exact asset allocation that maximizes this number.

### 4. Maximum Drawdown (MDD)
* **What it is:** The absolute worst-case historical scenario. It measures the largest single drop in portfolio value from a peak to a trough before a new peak is achieved.
* **How it's calculated:** The engine tracks the running maximum value of your portfolio over the historical timeframe. It compares the portfolio's value on any given day to that historical peak to find the steepest percentage decline.
* **Why it matters:** MDD is a psychological and practical limit. If a portfolio has an MDD of -45%, you have to ask yourself: *"Would I panic-sell if I lost 45% of my money?"* If the answer is yes, the portfolio is too risky for you, regardless of its Expected Return.

### 5. Conditional Value at Risk (CVaR)
* **What it is:** Also known as **Expected Shortfall**. While standard Value at Risk (VaR) tells you the *minimum* you might lose on a very bad day, CVaR tells you the *average amount* you will lose on those very bad days.
* **How it's calculated:** The backend sorts your daily returns from worst to best. It finds the 5th percentile worst day (e.g., the threshold where 95% of days are better). CVaR is the mathematical average of all the returns that fall *below* that 5th percentile line.
* **Why it matters:** It is a measure of "Tail Risk" (Black Swan events). In the 2008 financial crisis, many standard VaR models failed because they didn't measure the severity of the tail end. CVaR explicitly captures the magnitude of extreme market crashes.

### 6. Factor Exposure Betas (Market, Size, Value)
* **What it is:** A Fama-French style analysis that breaks down *why* your portfolio is generating returns by measuring its sensitivity to specific macroeconomic themes.
* **How it's calculated:** Using **Multiple Linear Regression** (`scipy.linalg.lstsq`), the engine plots your portfolio's returns against three proxy indices:
  * **Market Beta (`^GSPC`)**: If your Market Beta is `1.2`, your portfolio is 20% more volatile than the S&P 500.
  * **Size Beta (`^RUT`)**: Measures exposure to Small-Cap stocks (Russell 2000). A high number means your portfolio behaves like a volatile startup fund.
  * **Value Beta (`VTV`)**: Measures exposure to Value stocks (Vanguard Value ETF). A high number means your portfolio is heavily anchored in stable, dividend-paying legacy companies.
* **Why it matters:** Institutional investors use this to ensure they aren't accidentally taking on hidden risks. (e.g., You might think you are diversified, but a factor analysis might reveal you are actually 90% exposed to Small-Cap volatility).

### 7. Monte Carlo Percentiles (10th, 50th, 90th)
* **What it is:** A stochastic forecast of your portfolio's wealth 10 years into the future.
* **How it's calculated:** Using **Geometric Brownian Motion (GBM)**, the engine rolls virtual dice 1,000 times for every single trading day over the next 10 years, factoring in your portfolio's specific expected return (drift) and volatility (shock). 
* **Why it matters:** 
  * **10th Percentile (Bottom Line)**: The "Bear Market" scenario. There is a 90% statistical probability you will end up with *at least* this much money.
  * **50th Percentile (Middle Line)**: The "Base Case" or Median expectation.
  * **90th Percentile (Top Line)**: The "Bull Market" scenario. The optimistic upper bound of your wealth projection.
