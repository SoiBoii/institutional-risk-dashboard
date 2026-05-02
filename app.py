import json
from flask import Flask, request, jsonify, session
from flask_cors import CORS
import yfinance as yf
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from datetime import datetime, timedelta
import traceback
from textblob import TextBlob

from models import db, User, Portfolio, Transaction, Watchlist, AccountHistory
from flask_login import LoginManager, login_user, logout_user, login_required, current_user

app = Flask(__name__)
app.config['SECRET_KEY'] = 'cyberpunk-super-secret-key-2026'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///cyberpunk.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

CORS(app, supports_credentials=True)
db.init_app(app)

login_manager = LoginManager()
login_manager.init_app(app)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

with app.app_context():
    db.create_all()

TRADING_DAYS = 252
RISK_FREE_RATE = 0.02
BENCHMARK = '^GSPC'
SIZE_PROXY = '^RUT'
VALUE_PROXY = 'VTV'

# --- Auth & State Endpoints ---

@app.route('/register', methods=['POST'])
def register():
    data = request.json
    username = data.get('username')
    password = data.get('password')
    if not username or not password:
        return jsonify({'error': 'Username and password required'}), 400
    if User.query.filter_by(username=username).first():
        return jsonify({'error': 'Username already exists'}), 400
    
    user = User(username=username)
    user.set_password(password)
    db.session.add(user)
    db.session.commit()
    login_user(user)
    return jsonify({'message': 'Registered successfully', 'username': username})

@app.route('/login', methods=['POST'])
def login():
    data = request.json
    username = data.get('username')
    password = data.get('password')
    user = User.query.filter_by(username=username).first()
    if user and user.check_password(password):
        login_user(user)
        return jsonify({'message': 'Logged in successfully', 'username': username})
    return jsonify({'error': 'Invalid username or password'}), 401

@app.route('/logout', methods=['POST'])
@login_required
def logout():
    logout_user()
    return jsonify({'message': 'Logged out successfully'})

@app.route('/user_info', methods=['GET'])
def user_info():
    if current_user.is_authenticated:
        ports = Portfolio.query.filter_by(user_id=current_user.id).all()
        total_val = sum(p.total_value for p in ports)
        
        # Take snapshot
        today = datetime.utcnow().date()
        history = AccountHistory.query.filter_by(user_id=current_user.id, date=today).first()
        if history:
            history.total_value = total_val
        else:
            new_history = AccountHistory(user_id=current_user.id, date=today, total_value=total_val)
            db.session.add(new_history)
        db.session.commit()
        
        return jsonify({
            'logged_in': True, 
            'username': current_user.username, 
            'total_account_value': total_val,
            'tier': current_user.tier,
            'theme_color': current_user.theme_color
        })
    return jsonify({'logged_in': False})

@app.route('/save_portfolio', methods=['POST'])
@login_required
def save_portfolio():
    data = request.json
    name = data.get('portfolio_name')
    if not name:
        return jsonify({'error': 'Portfolio name required'}), 400
    
    config_json = json.dumps({
        'tickers': data.get('tickers', []),
        'weights': data.get('weights', []),
        'timeframe': data.get('timeframe', '1y')
    })
    
    portfolio = Portfolio(
        user_id=current_user.id,
        portfolio_name=name,
        total_value=float(data.get('total_value', 100000)),
        configuration=config_json
    )
    db.session.add(portfolio)
    db.session.commit()
    return jsonify({'message': 'Portfolio saved successfully', 'id': portfolio.id})

@app.route('/load_portfolios', methods=['GET'])
@login_required
def load_portfolios():
    ports = Portfolio.query.filter_by(user_id=current_user.id).all()
    res = []
    for p in ports:
        res.append({
            'id': p.id,
            'name': p.portfolio_name,
            'total_value': p.total_value,
            'config': json.loads(p.configuration)
        })
    return jsonify({'portfolios': res})


@app.route('/portfolio/<int:portfolio_id>/state', methods=['GET'])
@login_required
def get_portfolio_state(portfolio_id):
    p = Portfolio.query.get_or_404(portfolio_id)
    if p.user_id != current_user.id:
        return jsonify({'error': 'Unauthorized'}), 403
        
    transactions = Transaction.query.filter_by(portfolio_id=p.id).order_by(Transaction.date.asc()).all()
    holdings = {}
    cost_basis = 0.0
    
    tx_list = []
    for t in transactions:
        tx_list.insert(0, {
            'id': t.id, 'ticker': t.ticker, 'type': t.transaction_type,
            'quantity': t.quantity, 'price': t.price, 'date': t.date.strftime('%Y-%m-%d %H:%M')
        })
        
    for t in transactions:
        if t.ticker not in holdings:
            holdings[t.ticker] = {'qty': 0, 'cost': 0}
            
        if t.transaction_type == 'BUY':
            holdings[t.ticker]['qty'] += t.quantity
            holdings[t.ticker]['cost'] += (t.quantity * t.price)
            cost_basis += (t.quantity * t.price)
        else:
            if holdings[t.ticker]['qty'] > 0:
                avg_cost = holdings[t.ticker]['cost'] / holdings[t.ticker]['qty']
                holdings[t.ticker]['qty'] -= t.quantity
                holdings[t.ticker]['cost'] -= (t.quantity * avg_cost)
                cost_basis -= (t.quantity * avg_cost)
                
    holdings = {k: v for k, v in holdings.items() if v['qty'] > 0}
    tickers = list(holdings.keys())
    current_value = 0.0
    prices = {}
    
    if tickers:
        data = yf.download(tickers, period='5d')
        if not data.empty:
            col = 'Adj Close' if 'Adj Close' in data else 'Close'
            if isinstance(data.columns, pd.MultiIndex):
                df = data[col]
            else:
                df = data.to_frame() if isinstance(data, pd.Series) else data[col]
                
            for t in tickers:
                if t in df.columns or (isinstance(df, pd.Series) and t == tickers[0]):
                    series = df[t].dropna() if isinstance(df, pd.DataFrame) else df.dropna()
                    if not series.empty:
                        prices[t] = float(series.iloc[-1])
                            
    for t, h in holdings.items():
        current_value += h['qty'] * prices.get(t, 0)
        
    weights = []
    if current_value > 0:
        weights = [(holdings[t]['qty'] * prices.get(t, 0)) / current_value for t in tickers]
        
    unrealized_pnl = current_value - cost_basis
    
    # Update portfolio total value in db based on current market value
    if current_value > 0:
        p.total_value = current_value
        db.session.commit()
    
    return jsonify({
        'tickers': tickers,
        'weights': weights,
        'current_value': current_value,
        'cost_basis': cost_basis,
        'unrealized_pnl': unrealized_pnl,
        'transactions': tx_list
    })

@app.route('/portfolio/<int:portfolio_id>/trade', methods=['POST'])
@login_required
def execute_trade(portfolio_id):
    p = Portfolio.query.get_or_404(portfolio_id)
    if p.user_id != current_user.id:
        return jsonify({'error': 'Unauthorized'}), 403
    
    data = request.json
    ticker = data.get('ticker', '').upper()
    ttype = data.get('type', 'BUY').upper()
    qty = float(data.get('quantity', 0))
    
    if qty <= 0 or not ticker:
        return jsonify({'error': 'Invalid trade parameters'}), 400
        
    
    # Enforce standard limits
    if current_user.tier == 'Standard':
        existing_txs = Transaction.query.filter_by(portfolio_id=p.id).all()
        unique_tickers = set(t.ticker for t in existing_txs)
        if ticker not in unique_tickers and len(unique_tickers) >= 3:
            return jsonify({'error': 'UPGRADE_REQUIRED', 'message': 'Standard tier limits portfolios to 3 assets. Upgrade to Pro for unlimited.'}), 403

    yf_data = yf.download(ticker, period='5d')
    if yf_data.empty:
        return jsonify({'error': 'Ticker not found'}), 404
        
    col = 'Adj Close' if 'Adj Close' in yf_data else 'Close'
    df = yf_data[col]
    series = df[ticker].dropna() if isinstance(df, pd.DataFrame) else df.dropna()
    if series.empty:
        return jsonify({'error': 'Ticker price not available'}), 404
        
    price = float(series.iloc[-1])
    
    t = Transaction(portfolio_id=p.id, ticker=ticker, transaction_type=ttype, quantity=qty, price=price)
    db.session.add(t)
    db.session.commit()
    
    return jsonify({'message': 'Trade executed', 'price': price})

@app.route('/watchlist', methods=['GET', 'POST', 'DELETE'])
@login_required
def watchlist_api():
    if request.method == 'GET':
        items = Watchlist.query.filter_by(user_id=current_user.id).all()
        tickers = [i.ticker for i in items]
        alerts = []
        if tickers:
            data = yf.download(tickers, period='5d')
            if not data.empty:
                col = 'Adj Close' if 'Adj Close' in data else 'Close'
                df = data[col] if isinstance(data.columns, pd.MultiIndex) else (data.to_frame() if isinstance(data, pd.Series) else data[col])
                for t in tickers:
                    if t in df.columns or (isinstance(df, pd.Series) and t == tickers[0]):
                        series = df[t].dropna() if isinstance(df, pd.DataFrame) else df.dropna()
                        if len(series) >= 2:
                            cur = float(series.iloc[-1])
                            prev = float(series.iloc[-2])
                            pct = (cur - prev) / prev
                            alerts.append({'ticker': t, 'price': cur, 'change': pct})
        return jsonify({'watchlist': alerts})
        
    elif request.method == 'POST':
        ticker = request.json.get('ticker', '').upper()

        if not ticker:
            return jsonify({'error': 'Ticker required'}), 400
            
        if current_user.tier == 'Standard':
            wl_count = Watchlist.query.filter_by(user_id=current_user.id).count()
            if not Watchlist.query.filter_by(user_id=current_user.id, ticker=ticker).first() and wl_count >= 2:
                return jsonify({'error': 'UPGRADE_REQUIRED', 'message': 'Standard tier limits Watchlist to 2 assets. Upgrade to Pro for unlimited.'}), 403
                
        if not Watchlist.query.filter_by(user_id=current_user.id, ticker=ticker).first():
            w = Watchlist(user_id=current_user.id, ticker=ticker)
            db.session.add(w)
            db.session.commit()
        return jsonify({'message': 'Added to watchlist'})
        
    elif request.method == 'DELETE':
        ticker = request.json.get('ticker', '').upper()
        w = Watchlist.query.filter_by(user_id=current_user.id, ticker=ticker).first()
        if w:
            db.session.delete(w)
            db.session.commit()
        return jsonify({'message': 'Removed from watchlist'})


@app.route('/settings', methods=['POST'])
@login_required
def update_settings():
    color = request.json.get('theme_color')
    if color in ['cyan', 'green', 'purple']:
        current_user.theme_color = color
        db.session.commit()
        return jsonify({'message': 'Theme updated'})
    return jsonify({'error': 'Invalid theme'}), 400

@app.route('/leaderboard', methods=['GET'])
@login_required
def leaderboard():
    users = User.query.all()
    board = []
    for u in users:
        ports = Portfolio.query.filter_by(user_id=u.id).all()
        if not ports: continue
        total_val = sum(p.total_value for p in ports)
        # Assuming initial capital of 100k per portfolio created
        start_val = len(ports) * 100000.0
        ret = (total_val - start_val) / start_val if start_val > 0 else 0
        board.append({'username': u.username, 'return': ret * 100})
    board.sort(key=lambda x: x['return'], reverse=True)
    return jsonify({'leaderboard': board[:10]})

@app.route('/account_history', methods=['GET'])
@login_required
def account_history():
    history = AccountHistory.query.filter_by(user_id=current_user.id).order_by(AccountHistory.date.asc()).all()
    dates = [h.date.strftime('%Y-%m-%d') for h in history]
    values = [h.total_value for h in history]
    return jsonify({'dates': dates, 'values': values})

# --- Quant Data Endpoints ---


def fetch_data(tickers, timeframe):
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
    
    data = yf.download(all_tickers, start=start_date, end=end_date)
    
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
            data = data.to_frame()
            data.columns = all_tickers
        
    data = data.ffill().bfill()
    data = data.dropna(axis=1, how='all')
    
    latest_prices = data.iloc[-1].to_dict() if not data.empty else {}
    
    telemetry = {}
    for ticker in tickers:
        if ticker in data.columns:
            p_series = data[ticker].dropna()
            if not p_series.empty:
                cur = float(p_series.iloc[-1])
                prev = float(p_series.iloc[-2]) if len(p_series) > 1 else cur
                change = (cur - prev) / prev if prev else 0.0
                spark = p_series.iloc[-30:].tolist()
                full_history = p_series.tolist()
                full_dates = [d.strftime('%Y-%m-%d') for d in p_series.index]
                telemetry[ticker] = {
                    'current_price': cur,
                    'daily_change': change,
                    'sparkline': spark,
                    'history': full_history,
                    'dates': full_dates
                }

    returns = data.pct_change().dropna()
    
    missing_tickers = [t for t in tickers if t not in returns.columns]
    if missing_tickers:
        raise ValueError(f"Failed to fetch data for: {', '.join(missing_tickers)}")
        
    return returns, latest_prices, telemetry

def calculate_portfolio_performance(weights, returns):
    mean_returns = returns.mean()
    cov_matrix = returns.cov()
    
    port_return = np.sum(mean_returns * weights) * TRADING_DAYS
    port_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix * TRADING_DAYS, weights)))
    sharpe_ratio = (port_return - RISK_FREE_RATE) / port_volatility if port_volatility > 0 else 0
    return port_return, port_volatility, sharpe_ratio

def calculate_sortino(weights, returns):
    port_returns_daily = returns.dot(weights)
    mean_return = np.mean(port_returns_daily) * TRADING_DAYS
    
    downside_returns = port_returns_daily[port_returns_daily < 0]
    if len(downside_returns) > 0:
        downside_std = np.sqrt(np.mean(downside_returns**2)) * np.sqrt(TRADING_DAYS)
        sortino = (mean_return - RISK_FREE_RATE) / downside_std if downside_std > 0 else 0
    else:
        sortino = 0
    return sortino

def calculate_max_drawdown(weights, returns):
    port_returns_daily = returns.dot(weights)
    cumulative = (1 + port_returns_daily).cumprod()
    peak = cumulative.cummax()
    drawdown = (cumulative - peak) / peak
    return drawdown.min()

def calculate_var(weights, returns, confidence=0.05):
    port_returns_daily = returns.dot(weights)
    var = np.percentile(port_returns_daily, 100 * confidence)
    return var

def calculate_cvar(weights, returns, confidence=0.05):
    port_returns_daily = returns.dot(weights)
    var = np.percentile(port_returns_daily.dropna(), 100 * confidence)
    cvar = port_returns_daily[port_returns_daily <= var].mean()
    return cvar if not np.isnan(cvar) else var

def calculate_beta(weights, returns, benchmark_returns):
    if benchmark_returns is None or benchmark_returns.empty:
        return 1.0
    port_returns_daily = returns.dot(weights)
    bench_returns_daily = benchmark_returns.iloc[:, 0]
    
    aligned = pd.concat([port_returns_daily, bench_returns_daily], axis=1).dropna()
    if aligned.empty:
        return 1.0
        
    cov = np.cov(aligned.iloc[:, 0], aligned.iloc[:, 1])[0, 1]
    var_bench = np.var(aligned.iloc[:, 1])
    return cov / var_bench if var_bench > 0 else 1.0

def generate_kpis(weights, returns, benchmark_returns):
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
    port_returns_daily = returns.dot(weights)
    
    hist, bin_edges = np.histogram(port_returns_daily.dropna(), bins=50)
    
    rolling_returns = port_returns_daily.rolling(window=60).mean() * TRADING_DAYS
    rolling_std = port_returns_daily.rolling(window=60).std() * np.sqrt(TRADING_DAYS)
    rolling_std = rolling_std.replace(0, np.nan)
    rolling_sharpe = ((rolling_returns - RISK_FREE_RATE) / rolling_std).dropna()
    
    cumulative = (1 + port_returns_daily).cumprod()
    peak = cumulative.cummax()
    underwater = ((cumulative - peak) / peak).dropna()
    
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
    return -calculate_portfolio_performance(weights, returns)[2]

def portfolio_volatility(weights, returns):
    return calculate_portfolio_performance(weights, returns)[1]

def optimize_portfolio(returns, objective='max_sharpe'):
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
    orders = []
    for i, ticker in enumerate(tickers):
        curr_w = current_weights[i]
        opt_w = optimal_weights[i]
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
    data = request.json
    tickers = data.get('tickers', [])
    weights = data.get('weights', [])
    timeframe = data.get('timeframe', '1y')
    total_capital = float(data.get('total_capital', 100000))
    
    if not tickers:
        return jsonify({'error': 'No tickers provided.'}), 400
        
    if len(tickers) != len(weights):
        return jsonify({'error': 'The number of tickers and weights must match.'}), 400
        
    tickers = [str(t).strip().upper() for t in tickers]
    weights = np.array(weights, dtype=float)
    
    if np.sum(weights) == 0:
         return jsonify({'error': 'Weights cannot sum to zero.'}), 400
    weights = weights / np.sum(weights)
    
    try:
        full_returns, latest_prices, telemetry = fetch_data(tickers, timeframe)
        returns = full_returns[tickers]
        benchmark_returns = full_returns[[BENCHMARK]] if BENCHMARK in full_returns.columns else None
        
        user_kpis = generate_kpis(weights, returns, benchmark_returns)
        
        opt_weights = optimize_portfolio(returns, 'max_sharpe')
        opt_kpis = generate_kpis(opt_weights, returns, benchmark_returns)
        
        bench_kpis = None
        if benchmark_returns is not None and not benchmark_returns.empty:
            bench_weights = np.array([1.0])
            bench_kpis = generate_kpis(bench_weights, benchmark_returns, benchmark_returns)
            
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
            
        mc_results = monte_carlo_simulation(returns)
        corr_matrix = returns.corr().to_dict()
        
        deep_dive_data = generate_deep_dive_data(weights, returns, benchmark_returns)
        wealth_forecast = monte_carlo_wealth(weights, returns, initial_investment=total_capital)
        
        factor_exposure = calculate_factor_exposure(weights, returns, full_returns)
        rebalance_orders = calculate_rebalance(tickers, weights, opt_weights, latest_prices, total_capital)
        sentiment_news = get_sentiment_data(tickers)
        
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
            'sentiment_news': sentiment_news,
            'telemetry': telemetry
        }
        
        return jsonify(response)
        
    except ValueError as ve:
        return jsonify({'error': str(ve)}), 400
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': f'Server error: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(port=5050, debug=True)
