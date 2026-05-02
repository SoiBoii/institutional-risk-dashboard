import re

with open('app.py', 'r') as f:
    content = f.read()

# Fix imports
content = content.replace("from models import db, User, Portfolio", "from models import db, User, Portfolio, Transaction, Watchlist")

# Update user_info
new_user_info = """@app.route('/user_info', methods=['GET'])
def user_info():
    if current_user.is_authenticated:
        ports = Portfolio.query.filter_by(user_id=current_user.id).all()
        # For simplicity, we just sum saved total_values
        total_val = sum(p.total_value for p in ports)
        return jsonify({'logged_in': True, 'username': current_user.username, 'total_account_value': total_val})
    return jsonify({'logged_in': False})"""
content = re.sub(r"@app\.route\('/user_info', methods=\['GET'\]\).*?return jsonify\(\{'logged_in': False\}\)", new_user_info, content, flags=re.DOTALL)

# Add new endpoints before Quant Data Endpoints
new_endpoints = """
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
        
    yf_data = yf.download(ticker, period='5d')
    if yf_data.empty:
        return jsonify({'error': 'Ticker not found'}), 404
        
    col = 'Adj Close' if 'Adj Close' in yf_data else 'Close'
    series = yf_data[col].dropna()
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

# --- Quant Data Endpoints ---
"""

content = content.replace("# --- Quant Data Endpoints ---", new_endpoints)

with open('app.py', 'w') as f:
    f.write(content)
print("app.py updated")
