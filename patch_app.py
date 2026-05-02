import re

with open('app.py', 'r') as f:
    content = f.read()

# 1. Update imports
content = content.replace("from models import db, User, Portfolio, Transaction, Watchlist", "from models import db, User, Portfolio, Transaction, Watchlist, AccountHistory")

# 2. Update user_info and snapshot logic
new_user_info = """@app.route('/user_info', methods=['GET'])
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
    return jsonify({'logged_in': False})"""
content = re.sub(r"@app\.route\('/user_info', methods=\['GET'\]\).*?return jsonify\(\{'logged_in': False\}\)", new_user_info, content, flags=re.DOTALL)

# 3. Add Settings, Leaderboard, Account History, and modify limits
new_endpoints = """
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
"""

# Insert new endpoints before Quant Data Endpoints
content = content.replace("# --- Quant Data Endpoints ---", new_endpoints + "\n# --- Quant Data Endpoints ---")

# Apply limit in trade endpoint
trade_limit_check = """
    # Enforce standard limits
    if current_user.tier == 'Standard':
        existing_txs = Transaction.query.filter_by(portfolio_id=p.id).all()
        unique_tickers = set(t.ticker for t in existing_txs)
        if ticker not in unique_tickers and len(unique_tickers) >= 3:
            return jsonify({'error': 'UPGRADE_REQUIRED', 'message': 'Standard tier limits portfolios to 3 assets. Upgrade to Pro for unlimited.'}), 403

    yf_data = yf.download(ticker, period='5d')"""
content = content.replace("yf_data = yf.download(ticker, period='5d')", trade_limit_check)

# Apply limit in watchlist endpoint
wl_limit_check = """
        if not ticker:
            return jsonify({'error': 'Ticker required'}), 400
            
        if current_user.tier == 'Standard':
            wl_count = Watchlist.query.filter_by(user_id=current_user.id).count()
            if not Watchlist.query.filter_by(user_id=current_user.id, ticker=ticker).first() and wl_count >= 2:
                return jsonify({'error': 'UPGRADE_REQUIRED', 'message': 'Standard tier limits Watchlist to 2 assets. Upgrade to Pro for unlimited.'}), 403
                
        if not Watchlist.query.filter_by(user_id=current_user.id, ticker=ticker).first():"""
content = content.replace("""        if not ticker:
            return jsonify({'error': 'Ticker required'}), 400
        if not Watchlist.query.filter_by(user_id=current_user.id, ticker=ticker).first():""", wl_limit_check)

with open('app.py', 'w') as f:
    f.write(content)
