from flask import Flask, request, jsonify
from flask_cors import CORS
import yfinance as yf
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from datetime import datetime, timedelta
import traceback
from textblob import TextBlob

app = Flask(__name__)
CORS(app)

TRADING_DAYS = 252
RISK_FREE_RATE = 0.02
BENCHMARK = '^GSPC'
SIZE_PROXY = '^RUT'
VALUE_PROXY = 'VTV'

def fetch_data(tickers, timeframe):
    """
    Fetch historical daily 'Adj Close' prices for the list of tickers and the benchmark.
    Handles timeframe calculations and NaN values robustly.
    """
    end_date = datetime.today()
    if timeframe == '1y':
        days = 365
    elif timeframe == '3y':
        days = 365 * 3
    elif timeframe == '5y':
        days = 365 * 5
    else:
        days = 365
    start_date = end_date - timedelta(days=days)
    
    all_tickers = tickers + [BENCHMARK, SIZE_PROXY, VALUE_PROXY]
    
    # Download data
    data = yf.download(all_tickers, start=start_date, end=end_date)
    
    # Extract prices robustly
    if isinstance(data.columns, pd.MultiIndex):
        prices = pd.DataFrame()
        for ticker in all_tickers:
            if ('Adj Close', ticker) in data.columns and not data[('Adj Close', ticker)].isna().all():
                prices[ticker] = data[('Adj Close', ticker)]
            elif ('Close', ticker) in data.columns:
                prices[ticker] = data[('Close', ticker)]
        data = prices
    else:
        if 'Adj Close' in data:
            data = data['Adj Close']
        elif 'Close' in data:
            data = data['Close']
            
        if isinstance(data, pd.Series):
            # Fallback if only one ticker is somehow fetched
            data = data.to_frame()
            data.columns = all_tickers
        
    # Forward fill then backward fill for NaN values
    data = data.ffill().bfill()
    # Drop columns that are entirely NaN
    data = data.dropna(axis=1, how='all')
    
    latest_prices = data.iloc[-1].to_dict() if not data.empty else {}
    
    # Calculate daily percentage returns
    returns = data.pct_change().dropna()
    
    # Identify unresolvable tickers
    missing_tickers = [t for t in tickers if t not in returns.columns]
    if missing_tickers:
        raise ValueError(f"Failed to fetch data for: {', '.join(missing_tickers)}")
        
    return returns, latest_prices

def calculate_portfolio_performance(weights, returns):
    """Calculate annualized expected return, annualized volatility, and Sharpe Ratio."""
    mean_returns = returns.mean()
    cov_matrix = returns.cov()
    
    port_return = np.sum(mean_returns * weights) * TRADING_DAYS
    port_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix * TRADING_DAYS, weights)))
    sharpe_ratio = (port_return - RISK_FREE_RATE) / port_volatility if port_volatility > 0 else 0
    return port_return, port_volatility, sharpe_ratio

def calculate_sortino(weights, returns):
    """Calculate the Sortino Ratio (penalizes only downside volatility)."""
    port_returns_daily = returns.dot(weights)
    mean_return = np.mean(port_returns_daily) * TRADING_DAYS
    
    # Isolate negative returns
    downside_returns = port_returns_daily[port_returns_daily < 0]
    if len(downside_returns) > 0:
        downside_std = np.sqrt(np.mean(downside_returns**2)) * np.sqrt(TRADING_DAYS)
        sortino = (mean_return - RISK_FREE_RATE) / downside_std if downside_std > 0 else 0
    else:
        sortino = 0
    return sortino

def calculate_max_drawdown(weights, returns):
    """Calculate Maximum Drawdown (largest peak-to-trough drop)."""
    port_returns_daily = returns.dot(weights)
    cumulative = (1 + port_returns_daily).cumprod()
    peak = cumulative.cummax()
    drawdown = (cumulative - peak) / peak
    return drawdown.min()

def calculate_var(weights, returns, confidence=0.05):
    """Calculate Value at Risk (VaR) using historical simulation."""
    port_returns_daily = returns.dot(weights)
    var = np.percentile(port_returns_daily, 100 * confidence)
    return var

def calculate_cvar(weights, returns, confidence=0.05):
    """Calculate Conditional Value at Risk (Expected Shortfall)."""
    port_returns_daily = returns.dot(weights)
    var = np.percentile(port_returns_daily.dropna(), 100 * confidence)
    cvar = port_returns_daily[port_returns_daily <= var].mean()
    return cvar if not np.isnan(cvar) else var

def calculate_beta(weights, returns, benchmark_returns):
    """Calculate Portfolio Beta relative to the S&P 500 benchmark."""
    if benchmark_returns is None or benchmark_returns.empty:
        return 1.0
    port_returns_daily = returns.dot(weights)
    bench_returns_daily = benchmark_returns.iloc[:, 0]
    
    # Align the indices just in case
    aligned = pd.concat([port_returns_daily, bench_returns_daily], axis=1).dropna()
    if aligned.empty:
        return 1.0
        
    cov = np.cov(aligned.iloc[:, 0], aligned.iloc[:, 1])[0, 1]
    var_bench = np.var(aligned.iloc[:, 1])
    return cov / var_bench if var_bench > 0 else 1.0

def generate_kpis(weights, returns, benchmark_returns):
    """Aggregate all advanced KPIs into a dictionary."""
    ret, vol, sharpe = calculate_portfolio_performance(weights, returns)
    sortino = calculate_sortino(weights, returns)
    max_dd = calculate_max_drawdown(weights, returns)
    var = calculate_var(weights, returns)
    cvar = calculate_cvar(weights, returns)
    beta = calculate_beta(weights, returns, benchmark_returns)
    
    return {
        'return': float(ret),
        'volatility': float(vol),
        'sharpe_ratio': float(sharpe),
        'sortino_ratio': float(sortino),
        'max_drawdown': float(max_dd),
        'var_95': float(var),
        'cvar_95': float(cvar),
        'beta': float(beta)
    }

def generate_deep_dive_data(weights, returns, benchmark_returns):
    """Generate granular time-series and distributions for the Deep Dive Modals."""
    port_returns_daily = returns.dot(weights)
    
    # 1. Return Histogram (Distribution)
    hist, bin_edges = np.histogram(port_returns_daily.dropna(), bins=50)
    
    # 2. Rolling 60-day Sharpe Ratio
    rolling_returns = port_returns_daily.rolling(window=60).mean() * TRADING_DAYS
    rolling_std = port_returns_daily.rolling(window=60).std() * np.sqrt(TRADING_DAYS)
    rolling_std = rolling_std.replace(0, np.nan)
    rolling_sharpe = ((rolling_returns - RISK_FREE_RATE) / rolling_std).dropna()
    
    # 3. Max Drawdown "Underwater Curve"
    cumulative = (1 + port_returns_daily).cumprod()
    peak = cumulative.cummax()
    underwater = ((cumulative - peak) / peak).dropna()
    
    # 4. Rolling 60-day Beta and 90-day Correlation
    rolling_beta = pd.Series(dtype=float)
    rolling_correlation = pd.Series(dtype=float)
    if benchmark_returns is not None and not benchmark_returns.empty:
        bench_daily_ret = benchmark_returns.iloc[:, 0]
        aligned = pd.concat([port_returns_daily, bench_daily_ret], axis=1).dropna()
        if not aligned.empty:
            rolling_cov = aligned.iloc[:, 0].rolling(window=60).cov(aligned.iloc[:, 1])
            rolling_var = aligned.iloc[:, 1].rolling(window=60).var()
            rolling_var = rolling_var.replace(0, np.nan)
            rolling_beta = (rolling_cov / rolling_var).dropna()
            
            rolling_correlation = aligned.iloc[:, 0].rolling(window=90).corr(aligned.iloc[:, 1]).dropna()
            
    return {
        'histogram': {
            'bins': bin_edges[:-1].tolist(),
            'frequencies': hist.tolist()
        },
        'rolling_sharpe': {
            'dates': [d.strftime('%Y-%m-%d') for d in rolling_sharpe.index],
            'values': rolling_sharpe.tolist()
        },
        'underwater': {
            'dates': [d.strftime('%Y-%m-%d') for d in underwater.index],
            'values': underwater.tolist()
        },
        'rolling_beta': {
            'dates': [d.strftime('%Y-%m-%d') for d in rolling_beta.index],
            'values': rolling_beta.tolist()
        },
        'rolling_correlation': {
            'dates': [d.strftime('%Y-%m-%d') for d in rolling_correlation.index],
            'values': rolling_correlation.tolist()
        }
    }

def negative_sharpe_ratio(weights, returns):
    """Objective function to minimize for Max Sharpe portfolio."""
    return -calculate_portfolio_performance(weights, returns)[2]

def portfolio_volatility(weights, returns):
    """Objective function to minimize for Min Volatility portfolio."""
    return calculate_portfolio_performance(weights, returns)[1]

def optimize_portfolio(returns, objective='max_sharpe'):
    """Compute the mathematically optimal portfolio weights using SLSQP."""
    num_assets = len(returns.columns)
    args = (returns,)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for _ in range(num_assets))
    initial_guess = num_assets * [1. / num_assets]
    
    if objective == 'max_sharpe':
        result = minimize(negative_sharpe_ratio, initial_guess, args=args,
                          method='SLSQP', bounds=bounds, constraints=constraints)
    elif objective == 'min_volatility':
        result = minimize(portfolio_volatility, initial_guess, args=args,
                          method='SLSQP', bounds=bounds, constraints=constraints)
    return result.x

def calculate_factor_exposure(weights, returns, full_returns):
    """Multiple linear regression to find factor betas."""
    port_returns = returns.dot(weights)
    factors = [BENCHMARK, SIZE_PROXY, VALUE_PROXY]
    available_factors = [f for f in factors if f in full_returns.columns]
    
    if len(available_factors) < 3:
        return {'market': 1.0, 'size': 0.0, 'value': 0.0}
        
    X = full_returns[available_factors].values
    y = port_returns.values
    
    X = np.column_stack([np.ones(len(X)), X])
    try:
        coefs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        return {
            'market': float(coefs[1]),
            'size': float(coefs[2]),
            'value': float(coefs[3])
        }
    except:
        return {'market': 1.0, 'size': 0.0, 'value': 0.0}

def calculate_rebalance(tickers, current_weights, optimal_weights, latest_prices, total_capital):
    """Calculate the dollar and share amounts to execute."""
    orders = []
    for i, ticker in enumerate(tickers):
        curr_w = current_weights[i]
        opt_w = optimal_weights[i]
        # Handle tuple keys from yfinance multi-index if necessary
        price = 1.0
        for k, v in latest_prices.items():
            if ticker in str(k):
                price = v
                break
        
        target_value = total_capital * opt_w
        current_value = total_capital * curr_w
        dollar_flow = target_value - current_value
        shares_to_trade = dollar_flow / price if price > 0 else 0
        
        if abs(dollar_flow) > 1.0:
            orders.append({
                'ticker': ticker,
                'action': 'BUY' if dollar_flow > 0 else 'SELL',
                'shares': float(abs(shares_to_trade)),
                'amount': float(abs(dollar_flow)),
                'price': float(price)
            })
    return orders

def get_sentiment_data(tickers):
    """Fetch recent news and analyze sentiment with TextBlob."""
    news_feed = []
    for ticker in tickers:
        try:
            t = yf.Ticker(ticker)
            news = t.news[:3] if hasattr(t, 'news') else []
            for item in news:
                title = item.get('content', {}).get('title', '') or item.get('title', '')
                if not title: continue
                blob = TextBlob(title)
                polarity = blob.sentiment.polarity
                
                if polarity > 0.1: sentiment = 'BULL'
                elif polarity < -0.1: sentiment = 'BEAR'
                else: sentiment = 'NEUTRAL'
                    
                news_feed.append({
                    'ticker': ticker,
                    'headline': title,
                    'sentiment': sentiment,
                    'polarity': float(polarity)
                })
        except Exception:
            pass
    return news_feed

def monte_carlo_simulation(returns, num_portfolios=1000):
    """Generate 1000 random portfolios for the Efficient Frontier scatter plot."""
    num_assets = len(returns.columns)
    results = np.zeros((3, num_portfolios))
    
    for i in range(num_portfolios):
        weights = np.random.random(num_assets)
        weights /= np.sum(weights)
        
        ret, vol, sharpe = calculate_portfolio_performance(weights, returns)
        results[0, i] = vol
        results[1, i] = ret
        results[2, i] = sharpe
        
    return results

def monte_carlo_wealth(weights, returns, initial_investment=10000, years=10, num_simulations=1000):
    """Generate GBM Monte Carlo future wealth projection paths."""
    port_returns_daily = returns.dot(weights).dropna()
    if port_returns_daily.empty:
        return {}
        
    mu = port_returns_daily.mean() * TRADING_DAYS
    sigma = port_returns_daily.std() * np.sqrt(TRADING_DAYS)
    dt = 1 / TRADING_DAYS
    steps = int(years * TRADING_DAYS)
    
    random_shocks = np.random.normal((mu - 0.5 * sigma**2) * dt, sigma * np.sqrt(dt), (steps, num_simulations))
    price_paths = np.vstack([np.zeros(num_simulations), random_shocks]).cumsum(axis=0)
    wealth_paths = initial_investment * np.exp(price_paths)
    
    p10 = np.percentile(wealth_paths, 10, axis=1)
    p50 = np.percentile(wealth_paths, 50, axis=1)
    p90 = np.percentile(wealth_paths, 90, axis=1)
    
    step = max(1, steps // 100)
    labels = [f"Y{round(i/TRADING_DAYS, 1)}" for i in range(steps + 1)]
    
    return {
        'labels': labels[::step],
        'p10': p10[::step].tolist(),
        'p50': p50[::step].tolist(),
        'p90': p90[::step].tolist()
    }

@app.route('/analyze', methods=['POST'])
def analyze():
    """Main API Endpoint for Portfolio Risk & Analytics"""
    data = request.json
    tickers = data.get('tickers', [])
    weights = data.get('weights', [])
    timeframe = data.get('timeframe', '1y')
    total_capital = float(data.get('total_capital', 100000))
    
    if not tickers:
        return jsonify({'error': 'No tickers provided.'}), 400
        
    if len(tickers) != len(weights):
        return jsonify({'error': 'The number of tickers and weights must match.'}), 400
        
    # Standardize tickers and prepare weights array
    tickers = [str(t).strip().upper() for t in tickers]
    weights = np.array(weights, dtype=float)
    
    if np.sum(weights) == 0:
         return jsonify({'error': 'Weights cannot sum to zero.'}), 400
    weights = weights / np.sum(weights)
    
    try:
        # 1. Fetch historical data
        full_returns, latest_prices = fetch_data(tickers, timeframe)
        returns = full_returns[tickers]
        benchmark_returns = full_returns[[BENCHMARK]] if BENCHMARK in full_returns.columns else None
        
        # 2. Compute KPIs for User Portfolio
        user_kpis = generate_kpis(weights, returns, benchmark_returns)
        
        # 3. Compute KPIs for Optimal Portfolios
        opt_weights = optimize_portfolio(returns, 'max_sharpe')
        opt_kpis = generate_kpis(opt_weights, returns, benchmark_returns)
        
        # 4. Compute KPIs for the S&P 500 Benchmark
        bench_kpis = None
        if benchmark_returns is not None and not benchmark_returns.empty:
            bench_weights = np.array([1.0])
            bench_kpis = generate_kpis(bench_weights, benchmark_returns, benchmark_returns)
            
        # 5. Generate Cumulative Returns Backtest Time-series
        user_daily_ret = returns.dot(weights)
        opt_daily_ret = returns.dot(opt_weights)
        
        user_cum = (1 + user_daily_ret).cumprod() - 1
        opt_cum = (1 + opt_daily_ret).cumprod() - 1
        
        timeseries_backtest = {
            'dates': [d.strftime('%Y-%m-%d') for d in returns.index],
            'user': user_cum.tolist(),
            'optimal': opt_cum.tolist(),
            'benchmark': []
        }
        
        if benchmark_returns is not None and not benchmark_returns.empty:
            bench_daily_ret = benchmark_returns.iloc[:, 0]
            bench_cum = (1 + bench_daily_ret).cumprod() - 1
            timeseries_backtest['benchmark'] = bench_cum.tolist()
            
        # 6. Monte Carlo Efficient Frontier & Correlation Matrix
        mc_results = monte_carlo_simulation(returns)
        corr_matrix = returns.corr().to_dict()
        
        # 7. Generate Deep Dive Data
        deep_dive_data = generate_deep_dive_data(weights, returns, benchmark_returns)
        
        # 8. Generate Wealth Forecast
        wealth_forecast = monte_carlo_wealth(weights, returns)
        
        # 9. New Institutional Features
        factor_exposure = calculate_factor_exposure(weights, returns, full_returns)
        rebalance_orders = calculate_rebalance(tickers, weights, opt_weights, latest_prices, total_capital)
        sentiment_news = get_sentiment_data(tickers)
        
        # Assemble Response
        response = {
            'kpis': {
                'user': user_kpis,
                'optimal': opt_kpis,
                'benchmark': bench_kpis
            },
            'weights': {
                'user': {tickers[i]: float(weights[i]) for i in range(len(tickers))},
                'optimal': {tickers[i]: float(opt_weights[i]) for i in range(len(tickers))}
            },
            'correlation_matrix': corr_matrix,
            'efficient_frontier': {
                'volatilities': mc_results[0].tolist(),
                'returns': mc_results[1].tolist()
            },
            'timeseries_backtest': timeseries_backtest,
            'deep_dive_data': deep_dive_data,
            'wealth_forecast': wealth_forecast,
            'factor_exposure': factor_exposure,
            'rebalance_orders': rebalance_orders,
            'sentiment_news': sentiment_news
        }
        
        return jsonify(response)
        
    except ValueError as ve:
        return jsonify({'error': str(ve)}), 400
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': f'Server error: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(port=5000, debug=True)
