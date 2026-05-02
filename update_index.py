import re

with open('/Users/saithejas/Desktop/FINAL_PROJECT_MCA/index.html', 'r') as f:
    content = f.read()

# 1. Add html2pdf and CSS animations
head_additions = """
    <!-- html2pdf for Tear Sheet -->
    <script src="https://cdnjs.cloudflare.com/ajax/libs/html2pdf.js/0.10.1/html2pdf.bundle.min.js"></script>
    
    <style>
        /* Keep existing styles */
        body { background-color: #09090b; color: #e5e7eb; font-family: 'Inter', sans-serif;
            background-image: linear-gradient(rgba(0, 243, 255, 0.03) 1px, transparent 1px),
                              linear-gradient(90deg, rgba(0, 243, 255, 0.03) 1px, transparent 1px);
            background-size: 30px 30px;
        }
        .font-mono { font-family: 'Space Mono', monospace; }
        .cyber-panel { background: rgba(9, 9, 11, 0.85); backdrop-filter: blur(12px); border: 1px solid rgba(0, 243, 255, 0.15); box-shadow: 0 0 15px rgba(0, 243, 255, 0.02); transition: all 0.2s ease; }
        .cyber-panel:hover { border-color: rgba(0, 243, 255, 0.4); box-shadow: 0 0 20px rgba(0, 243, 255, 0.1); }
        .cyber-input { background: rgba(0, 0, 0, 0.6); border: 1px solid rgba(0, 243, 255, 0.3); color: #00f3ff; transition: all 0.2s; }
        .cyber-input:focus { outline: none; border-color: #00f3ff; box-shadow: 0 0 10px rgba(0, 243, 255, 0.3); }
        .cyber-btn { background: rgba(0, 243, 255, 0.05); border: 1px solid #00f3ff; color: #00f3ff; transition: all 0.2s; }
        .cyber-btn:hover { background: rgba(0, 243, 255, 0.2); box-shadow: 0 0 15px rgba(0, 243, 255, 0.4); }
        .loader { border: 2px solid rgba(0, 243, 255, 0.1); border-top: 2px solid #00f3ff; border-radius: 50%; width: 16px; height: 16px; animation: spin 1s linear infinite; }
        @keyframes spin { 100% { transform: rotate(360deg); } }
        
        /* New Animations */
        @keyframes bootSequence {
            0% { filter: blur(10px) brightness(0.5); opacity: 0; transform: scale(0.98); }
            50% { filter: blur(2px) brightness(1.2); opacity: 0.8; }
            100% { filter: blur(0px) brightness(1); opacity: 1; transform: scale(1); }
        }
        .animate-boot { animation: bootSequence 1s cubic-bezier(0.1, 0.8, 0.1, 1) forwards; }
        .animate-pulse-red { animation: pulseRed 2s infinite; }
        @keyframes pulseRed { 0% { box-shadow: 0 0 0 0 rgba(255, 0, 60, 0.7); border-color: rgba(255, 0, 60, 1); } 70% { box-shadow: 0 0 0 10px rgba(255, 0, 60, 0); border-color: rgba(255, 0, 60, 0.3); } 100% { box-shadow: 0 0 0 0 rgba(255, 0, 60, 0); border-color: rgba(255, 0, 60, 1); } }
    </style>
"""
content = re.sub(r'<style>.*?</style>', head_additions, content, flags=re.DOTALL)

# 2. Update Nav Bar to Command Center
old_nav = """        <div class="flex flex-col md:flex-row justify-between items-start md:items-center mb-6 border-b border-cyber-cyan/20 pb-4 gap-4">
            <div>
                <h1 class="text-3xl font-bold tracking-widest text-white flex items-center gap-3 font-mono uppercase drop-shadow-[0_0_8px_rgba(255,255,255,0.2)]">
                    <span class="text-cyber-cyan drop-shadow-[0_0_8px_rgba(0,243,255,0.4)]">sys.</span>Quant_Engine
                </h1>
                <p class="mt-2 text-cyber-cyan/60 font-mono text-xs uppercase tracking-widest">>> Predictive Analytics & Risk Modeling HUD</p>
            </div>
            
            <div id="auth-nav" class="flex gap-4 font-mono text-sm items-center">
                <button onclick="openAuthModal('login')" class="text-cyber-cyan hover:text-white transition-colors">[ LOGIN ]</button>
                <button onclick="openAuthModal('register')" class="text-cyber-purple hover:text-white transition-colors">[ REGISTER ]</button>
            </div>
            
            <div id="user-nav" class="hidden flex-wrap justify-end items-center gap-3 font-mono text-sm w-full md:w-auto">
                <span class="text-cyber-cyan text-xs">HANDLE: <span id="current-username" class="text-white drop-shadow-[0_0_5px_rgba(255,255,255,0.5)]"></span></span>
                <button onclick="savePortfolio()" class="cyber-btn px-2 py-1 text-[10px]">[ SAVE_LAYOUT ]</button>
                <select id="portfolio-select" onchange="loadSelectedPortfolio()" class="cyber-input px-2 py-1 text-[10px] max-w-[140px] appearance-none cursor-pointer">
                    <option value="">-- LOAD_LAYOUT --</option>
                </select>
                <button onclick="logout()" class="text-cyber-red hover:text-white transition-colors ml-2 text-xs">[ EXIT ]</button>
            </div>
        </div>"""

new_nav = """        <div class="flex flex-col md:flex-row justify-between items-start md:items-center mb-6 border-b border-cyber-cyan/20 pb-4 gap-4">
            <div>
                <h1 class="text-3xl font-bold tracking-widest text-white flex items-center gap-3 font-mono uppercase drop-shadow-[0_0_8px_rgba(255,255,255,0.2)]">
                    <span class="text-cyber-cyan drop-shadow-[0_0_8px_rgba(0,243,255,0.4)]">sys.</span>Quant_Engine
                </h1>
                <p class="mt-2 text-cyber-cyan/60 font-mono text-xs uppercase tracking-widest">>> Institutional Risk & Execution HUD</p>
            </div>
            
            <div id="auth-nav" class="flex gap-4 font-mono text-sm items-center">
                <button onclick="openAuthModal('login')" class="text-cyber-cyan hover:text-white transition-colors">[ LOGIN ]</button>
                <button onclick="openAuthModal('register')" class="text-cyber-purple hover:text-white transition-colors">[ REGISTER ]</button>
            </div>
            
            <div id="user-nav" class="hidden flex-wrap justify-end items-center gap-3 font-mono text-sm w-full md:w-auto">
                <div class="flex flex-col items-end mr-2 border-r border-cyber-cyan/30 pr-3">
                    <span class="text-cyber-cyan text-[10px]">SYS_ADMIN: <span id="current-username" class="text-white drop-shadow-[0_0_5px_rgba(255,255,255,0.5)]"></span></span>
                    <span class="text-cyber-green text-xs font-bold mt-1">TOTAL_ACCT: $<span id="total-account-value">0.00</span></span>
                </div>
                <button onclick="exportPDF()" class="cyber-btn px-2 py-1 text-[10px] border-cyber-purple text-cyber-purple hover:bg-cyber-purple/20 hover:shadow-[0_0_15px_rgba(188,19,254,0.4)]">[ PDF_TEARSHEET ]</button>
                <button onclick="savePortfolio()" class="cyber-btn px-2 py-1 text-[10px]">[ NEW_PORTFOLIO ]</button>
                <select id="portfolio-select" onchange="loadSelectedPortfolio()" class="cyber-input px-2 py-1 text-[10px] max-w-[140px] appearance-none cursor-pointer bg-[#09090b]">
                    <option value="">-- ACTIVE_PORTFOLIO --</option>
                </select>
                <button onclick="logout()" class="text-cyber-red hover:text-white transition-colors ml-2 text-xs">[ EXIT ]</button>
            </div>
        </div>"""
content = content.replace(old_nav, new_nav)

# 3. Restructure layout for Watchlist Sidebar
# Wrap controls, dashboard-content, and empty state in a grid to accommodate sidebar
old_main_wrap = """        <!-- Controls -->
        <div class="cyber-panel p-4 rounded-none flex flex-wrap items-end gap-4 w-full mb-8">"""
new_main_wrap = """        <div class="flex flex-col lg:flex-row gap-6 w-full">
            <!-- Left Main Column -->
            <div class="flex-1 flex flex-col min-w-0">
                <!-- Controls -->
                <div class="cyber-panel p-4 rounded-none flex flex-wrap items-end gap-4 w-full mb-8">"""
content = content.replace(old_main_wrap, new_main_wrap)

old_empty_state_end = """        </div>
    </div>

    <!-- MAIN APP SCRIPT -->"""
new_empty_state_end = """        </div>
            </div> <!-- End Left Main Column -->

            <!-- Right Sidebar: Ledger & Watchlist -->
            <div id="right-sidebar" class="w-full lg:w-[350px] flex-col gap-6 hidden">
                <!-- Trade Blotter / Ledger -->
                <div class="cyber-panel p-4 flex flex-col h-[400px]">
                    <h2 class="text-xs font-mono text-cyber-purple tracking-widest uppercase mb-3 border-b border-cyber-purple/30 pb-2 drop-shadow-[0_0_5px_rgba(188,19,254,0.5)]">
                        >> TRADE_BLOTTER
                    </h2>
                    
                    <div class="flex gap-2 mb-3">
                        <select id="trade-type" class="cyber-input text-[10px] px-2 py-1 w-20">
                            <option value="BUY">BUY</option>
                            <option value="SELL">SELL</option>
                        </select>
                        <input type="text" id="trade-ticker" placeholder="TICKER" class="cyber-input text-[10px] px-2 py-1 flex-1 uppercase">
                        <input type="number" id="trade-qty" placeholder="QTY" class="cyber-input text-[10px] px-2 py-1 w-16">
                        <button onclick="executeTrade()" class="cyber-btn text-[10px] px-3 py-1 bg-cyber-purple/10 border-cyber-purple text-cyber-purple hover:bg-cyber-purple/30">EXEC</button>
                    </div>

                    <div class="flex justify-between text-[10px] font-mono text-cyber-cyan/70 mb-2">
                        <span>Basis: $<span id="ledger-basis">0.00</span></span>
                        <span>PnL: <span id="ledger-pnl" class="text-white">0.00</span></span>
                    </div>

                    <div class="flex-1 overflow-y-auto min-h-0 border border-cyber-cyan/10 bg-black/50 p-2">
                        <table class="w-full text-left font-mono text-[10px]">
                            <tbody id="ledger-tbody" class="text-white">
                                <tr><td class="text-center text-cyber-cyan/50 py-4">NO_TRANSACTIONS</td></tr>
                            </tbody>
                        </table>
                    </div>
                </div>

                <!-- Watchlist Comm-Link -->
                <div class="cyber-panel p-4 flex flex-col flex-1 min-h-[300px]">
                    <h2 class="text-xs font-mono text-cyber-cyan tracking-widest uppercase mb-3 border-b border-cyber-cyan/30 pb-2">
                        <span class="w-2 h-2 inline-block rounded-full bg-cyber-red animate-pulse mr-1"></span>
                        >> WATCHLIST_LINK
                    </h2>
                    
                    <div class="flex gap-2 mb-3">
                        <input type="text" id="wl-ticker" placeholder="TICKER" class="cyber-input text-[10px] px-2 py-1 flex-1 uppercase">
                        <button onclick="addToWatchlist()" class="cyber-btn text-[10px] px-3 py-1">+</button>
                    </div>

                    <div id="watchlist-container" class="flex flex-col gap-2 overflow-y-auto min-h-0 pr-1">
                        <!-- Populated by JS -->
                    </div>
                </div>
            </div> <!-- End Right Sidebar -->
        </div> <!-- End Flex Wrapper -->
    </div>

    <!-- MAIN APP SCRIPT -->"""
content = content.replace(old_empty_state_end, new_empty_state_end)

# 4. JS Logic Patches
# Add active portfolio state tracking
content = content.replace("let portfoliosCache = [];", "let portfoliosCache = [];\n        let currentPortfolioId = null;\n        let rightSidebar = document.getElementById('right-sidebar');")

# Modify checkAuth
old_checkauth = """        async function checkAuth() {
            try {
                const res = await fetch(`${API_URL}/user_info`);
                const data = await res.json();
                if(data.logged_in) {
                    document.getElementById('auth-nav').classList.add('hidden');
                    document.getElementById('user-nav').classList.remove('hidden');
                    document.getElementById('user-nav').classList.add('flex');
                    document.getElementById('current-username').textContent = data.username;
                    loadPortfoliosDropdown();
                } else {
                    document.getElementById('auth-nav').classList.remove('hidden');
                    document.getElementById('user-nav').classList.add('hidden');
                    document.getElementById('user-nav').classList.remove('flex');
                }
            } catch(e) {
                console.error('Check auth failed', e);
            }
        }"""
new_checkauth = """        async function checkAuth() {
            try {
                const res = await fetch(`${API_URL}/user_info`);
                const data = await res.json();
                if(data.logged_in) {
                    document.getElementById('auth-nav').classList.add('hidden');
                    document.getElementById('user-nav').classList.remove('hidden');
                    document.getElementById('user-nav').classList.add('flex');
                    document.getElementById('current-username').textContent = data.username;
                    document.getElementById('total-account-value').textContent = (data.total_account_value || 0).toLocaleString(undefined, {minimumFractionDigits: 2});
                    
                    // Trigger Boot Animation
                    document.body.classList.remove('animate-boot');
                    void document.body.offsetWidth; // trigger reflow
                    document.body.classList.add('animate-boot');
                    
                    rightSidebar.classList.remove('hidden');
                    rightSidebar.classList.add('flex');
                    
                    loadPortfoliosDropdown();
                    fetchWatchlist();
                } else {
                    document.getElementById('auth-nav').classList.remove('hidden');
                    document.getElementById('user-nav').classList.add('hidden');
                    document.getElementById('user-nav').classList.remove('flex');
                    rightSidebar.classList.add('hidden');
                    rightSidebar.classList.remove('flex');
                    currentPortfolioId = null;
                }
            } catch(e) {
                console.error('Check auth failed', e);
            }
        }"""
content = content.replace(old_checkauth, new_checkauth)

# Add loadSelectedPortfolio logic updates to fetch state
old_loadport = """        function loadSelectedPortfolio() {
            const select = document.getElementById('portfolio-select');
            const id = select.value;
            if(!id) return;
            const port = portfoliosCache.find(p => p.id == id);
            if(port) {
                document.getElementById('tickers').value = port.config.tickers.join(', ');
                document.getElementById('weights').value = port.config.weights.join(', ');
                document.getElementById('timeframe').value = port.config.timeframe || '1y';
                document.getElementById('total_capital').value = port.total_value || 100000;
                showToast(`LAYOUT_LOADED: ${port.name}`, 'success');
                // Optional: auto-execute after load
                document.getElementById('run-btn').click();
            }
            select.value = "";
        }"""
new_loadport = """        async function loadSelectedPortfolio() {
            const select = document.getElementById('portfolio-select');
            const id = select.value;
            if(!id) return;
            currentPortfolioId = id;
            
            const port = portfoliosCache.find(p => p.id == id);
            showToast(`PORTFOLIO_ACTIVE: ${port.name}`, 'success');
            
            await refreshPortfolioState();
        }
        
        async function refreshPortfolioState() {
            if(!currentPortfolioId) return;
            try {
                const res = await fetch(`${API_URL}/portfolio/${currentPortfolioId}/state`);
                const data = await res.json();
                if(!res.ok) throw new Error(data.error);
                
                // Update UI Ledger
                document.getElementById('ledger-basis').textContent = data.cost_basis.toLocaleString(undefined, {minimumFractionDigits: 2});
                const pnlEl = document.getElementById('ledger-pnl');
                pnlEl.textContent = (data.unrealized_pnl >= 0 ? '+' : '') + data.unrealized_pnl.toLocaleString(undefined, {minimumFractionDigits: 2});
                pnlEl.className = data.unrealized_pnl >= 0 ? 'text-cyber-green' : 'text-cyber-red';
                
                const tbody = document.getElementById('ledger-tbody');
                if(data.transactions.length === 0) {
                    tbody.innerHTML = '<tr><td class="text-center text-cyber-cyan/50 py-4">NO_TRANSACTIONS</td></tr>';
                } else {
                    tbody.innerHTML = data.transactions.map(t => {
                        const color = t.type === 'BUY' ? 'text-cyber-green' : 'text-cyber-red';
                        return `
                            <tr class="border-b border-cyber-cyan/10 hover:bg-white/5">
                                <td class="py-1 text-cyber-cyan">${t.date.split(' ')[0].substring(5)}</td>
                                <td class="py-1 font-bold ${color}">${t.type}</td>
                                <td class="py-1">${t.ticker}</td>
                                <td class="py-1 text-right">${t.quantity}</td>
                                <td class="py-1 text-right">$${t.price.toFixed(2)}</td>
                            </tr>
                        `;
                    }).join('');
                }
                
                // Update Run Inputs
                if(data.tickers.length > 0) {
                    document.getElementById('tickers').value = data.tickers.join(', ');
                    document.getElementById('weights').value = data.weights.map(w => w.toFixed(4)).join(', ');
                    document.getElementById('total_capital').value = data.current_value.toFixed(2);
                    
                    // Auto Execute Quant Engine
                    document.getElementById('run-btn').click();
                } else {
                    // Empty portfolio
                    document.getElementById('tickers').value = '';
                    document.getElementById('weights').value = '';
                    document.getElementById('empty-state').classList.remove('hidden');
                    document.getElementById('dashboard-content').classList.add('hidden');
                    document.getElementById('dashboard-content').classList.remove('flex');
                    destroyAllCharts();
                }
                
                // Refresh total account value
                fetch(`${API_URL}/user_info`).then(r=>r.json()).then(d => {
                    document.getElementById('total-account-value').textContent = (d.total_account_value || 0).toLocaleString(undefined, {minimumFractionDigits: 2});
                });
                
            } catch(e) {
                showToast(e.message, 'error');
            }
        }
        
        async function executeTrade() {
            if(!currentPortfolioId) return showToast('NO_ACTIVE_PORTFOLIO', 'error');
            const ticker = document.getElementById('trade-ticker').value.toUpperCase();
            const type = document.getElementById('trade-type').value;
            const qty = document.getElementById('trade-qty').value;
            
            try {
                const res = await fetch(`${API_URL}/portfolio/${currentPortfolioId}/trade`, {
                    method: 'POST', headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({ticker, type, quantity: qty})
                });
                const data = await res.json();
                if(!res.ok) throw new Error(data.error);
                showToast(`TRADE_EXEC: ${type} ${qty} ${ticker} @ $${data.price.toFixed(2)}`, 'success');
                document.getElementById('trade-qty').value = '';
                document.getElementById('trade-ticker').value = '';
                await refreshPortfolioState();
            } catch(e) {
                showToast(e.message, 'error');
            }
        }
        
        async function fetchWatchlist() {
            try {
                const res = await fetch(`${API_URL}/watchlist`);
                const data = await res.json();
                const container = document.getElementById('watchlist-container');
                container.innerHTML = '';
                
                data.watchlist.forEach(w => {
                    const isAlert = w.change < -0.02; // dropped more than 2%
                    const borderClass = isAlert ? 'border-cyber-red animate-pulse-red' : 'border-cyber-cyan/20 border';
                    const colorClass = w.change >= 0 ? 'text-cyber-green' : 'text-cyber-red';
                    const sign = w.change >= 0 ? '+' : '';
                    
                    container.innerHTML += `
                        <div class="cyber-panel p-2 flex justify-between items-center ${borderClass}">
                            <span class="font-mono font-bold text-cyber-cyan text-sm">${w.ticker}</span>
                            <div class="text-right">
                                <div class="text-xs text-white">$${w.price.toFixed(2)}</div>
                                <div class="text-[10px] ${colorClass}">${sign}${(w.change*100).toFixed(2)}%</div>
                            </div>
                            <button onclick="removeFromWatchlist('${w.ticker}')" class="text-cyber-cyan/50 hover:text-cyber-red ml-2 text-xs">[X]</button>
                        </div>
                    `;
                });
            } catch(e) {}
        }
        
        async function addToWatchlist() {
            const ticker = document.getElementById('wl-ticker').value.toUpperCase();
            if(!ticker) return;
            try {
                await fetch(`${API_URL}/watchlist`, {
                    method: 'POST', headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({ticker})
                });
                document.getElementById('wl-ticker').value = '';
                fetchWatchlist();
            } catch(e) {}
        }
        
        async function removeFromWatchlist(ticker) {
            try {
                await fetch(`${API_URL}/watchlist`, {
                    method: 'DELETE', headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({ticker})
                });
                fetchWatchlist();
            } catch(e) {}
        }
        
        function exportPDF() {
            const element = document.getElementById('dashboard-content');
            if(element.classList.contains('hidden')) {
                return showToast('NO_DATA_TO_EXPORT', 'error');
            }
            showToast('GENERATING_TEARSHEET...', 'success');
            
            // Temporary styles for PDF
            element.style.background = '#09090b';
            element.style.padding = '20px';
            
            html2pdf().set({
                margin: 10,
                filename: 'Institutional_TearSheet.pdf',
                image: { type: 'jpeg', quality: 0.98 },
                html2canvas: { scale: 2, useCORS: true, logging: false },
                jsPDF: { unit: 'mm', format: 'a4', orientation: 'landscape' }
            }).from(element).save().then(() => {
                element.style.background = 'transparent';
                element.style.padding = '0';
                showToast('TEARSHEET_EXPORTED', 'success');
            });
        }
"""
content = content.replace(old_loadport, new_loadport)

with open('/Users/saithejas/Desktop/FINAL_PROJECT_MCA/index.html', 'w') as f:
    f.write(content)
print("index.html updated")
