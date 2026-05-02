import re

with open('/Users/saithejas/Desktop/FINAL_PROJECT_MCA/index.html', 'r') as f:
    content = f.read()

# 1. Modals: Add Rankings, Settings, and Upgrade Modals
modals_html = """
    <!-- Upgrade Modal -->
    <div id="upgrade-modal" class="fixed inset-0 z-[200] bg-black/90 backdrop-blur-sm hidden flex items-center justify-center p-4">
        <div class="cyber-panel p-8 max-w-md w-full border-cyber-red/50 shadow-[0_0_30px_rgba(255,0,60,0.3)]">
            <h2 class="text-2xl font-bold text-cyber-red uppercase font-mono mb-4 flex items-center gap-2">
                <span class="animate-pulse-red w-3 h-3 bg-cyber-red rounded-full"></span> ACCESS_DENIED
            </h2>
            <p id="upgrade-message" class="text-white mb-6 font-mono text-sm"></p>
            <div class="flex gap-4">
                <button onclick="document.getElementById('upgrade-modal').classList.add('hidden')" class="cyber-btn flex-1 py-2 font-mono">DISMISS</button>
                <button class="flex-1 py-2 font-mono bg-cyber-red text-white hover:bg-cyber-red/80 transition-colors border border-cyber-red shadow-[0_0_15px_rgba(255,0,60,0.5)]">UPGRADE_PRO</button>
            </div>
        </div>
    </div>

    <!-- Leaderboard Modal -->
    <div id="leaderboard-modal" class="fixed inset-0 z-[150] bg-black/90 backdrop-blur-sm hidden flex items-center justify-center p-4">
        <div class="cyber-panel p-8 max-w-2xl w-full">
            <h2 class="text-2xl font-bold text-cyber-cyan uppercase font-mono mb-6 border-b border-cyber-cyan/30 pb-2">
                >> SHADOW_SERVER_RANKINGS
            </h2>
            <div class="overflow-y-auto max-h-[60vh] border border-cyber-cyan/10 p-2">
                <table class="w-full text-left font-mono">
                    <thead>
                        <tr class="text-cyber-cyan/50 text-xs border-b border-cyber-cyan/20">
                            <th class="pb-2 pl-2">RANK</th>
                            <th class="pb-2">HANDLE</th>
                            <th class="pb-2 text-right pr-2">RETURN</th>
                        </tr>
                    </thead>
                    <tbody id="leaderboard-tbody" class="text-white text-sm">
                    </tbody>
                </table>
            </div>
            <button onclick="document.getElementById('leaderboard-modal').classList.add('hidden')" class="mt-6 cyber-btn px-6 py-2 font-mono">CLOSE</button>
        </div>
    </div>

    <!-- Settings Modal -->
    <div id="settings-modal" class="fixed inset-0 z-[150] bg-black/90 backdrop-blur-sm hidden flex items-center justify-center p-4">
        <div class="cyber-panel p-8 max-w-sm w-full">
            <h2 class="text-xl font-bold text-cyber-cyan uppercase font-mono mb-6 border-b border-cyber-cyan/30 pb-2">
                >> UI_PREFERENCES
            </h2>
            <div class="flex flex-col gap-4 font-mono text-sm">
                <p class="text-cyber-cyan/70">SELECT THEME_COLOR:</p>
                <button onclick="updateTheme('cyan')" class="py-2 border border-[#00f3ff] text-[#00f3ff] hover:bg-[#00f3ff]/20">NEON_CYAN</button>
                <button onclick="updateTheme('green')" class="py-2 border border-[#39ff14] text-[#39ff14] hover:bg-[#39ff14]/20">MATRIX_GREEN</button>
                <button onclick="updateTheme('purple')" class="py-2 border border-[#bc13fe] text-[#bc13fe] hover:bg-[#bc13fe]/20">SYNTH_PURPLE</button>
            </div>
            <button onclick="document.getElementById('settings-modal').classList.add('hidden')" class="mt-6 cyber-btn w-full py-2 font-mono text-cyber-cyan/50">CLOSE</button>
        </div>
    </div>
"""
content = content.replace("<!-- Empty State -->", modals_html + "\n        <!-- Empty State -->")

# 2. Update Nav Bar to include Settings & Leaderboard
nav_updates = """
                <button onclick="openLeaderboard()" class="cyber-btn px-2 py-1 text-[10px] text-cyber-green border-cyber-green hover:bg-cyber-green/20">[ RANKINGS ]</button>
                <button onclick="document.getElementById('settings-modal').classList.remove('hidden')" class="text-cyber-cyan/50 hover:text-white transition-colors ml-1 text-xs">⚙️</button>
                <button onclick="exportPDF()" class="cyber-btn px-2 py-1 text-[10px] border-cyber-purple text-cyber-purple hover:bg-cyber-purple/20 hover:shadow-[0_0_15px_rgba(188,19,254,0.4)]">[ PDF_TEARSHEET ]</button>
"""
content = content.replace("""<button onclick="exportPDF()" class="cyber-btn px-2 py-1 text-[10px] border-cyber-purple text-cyber-purple hover:bg-cyber-purple/20 hover:shadow-[0_0_15px_rgba(188,19,254,0.4)]">[ PDF_TEARSHEET ]</button>""", nav_updates)

# 3. Add Account History Chart to Main Dashboard
history_chart = """
                    <!-- Account History -->
                    <div class="cyber-panel p-4 flex flex-col min-h-0 hidden" id="account-history-panel">
                        <h2 class="text-sm font-mono text-white tracking-widest uppercase mb-4 border-b border-white/20 pb-2">
                            >> EQUITY_CURVE
                        </h2>
                        <div class="flex-1 relative min-h-0">
                            <canvas id="accountHistoryChart"></canvas>
                        </div>
                    </div>
                    
                    <div class="cyber-panel p-4 flex flex-col min-h-0">
"""
content = content.replace("""<div class="cyber-panel p-4 flex flex-col min-h-0">""", history_chart)

# 4. JS logic for themes, modals, catching UPGRADE_REQUIRED, rendering history
js_updates = """
        let userThemeColor = 'cyan';
        const themeMap = {
            'cyan': '#00f3ff',
            'green': '#39ff14',
            'purple': '#bc13fe'
        };

        function applyTheme(colorName) {
            const hex = themeMap[colorName] || '#00f3ff';
            userThemeColor = colorName;
            Chart.defaults.color = hex;
            Chart.defaults.scale.ticks.color = hex + '99';
            Chart.defaults.plugins.tooltip.titleColor = hex;
            Chart.defaults.plugins.tooltip.borderColor = hex;
            mainChartOptions.plugins.tooltip.titleColor = hex;
            mainChartOptions.plugins.tooltip.borderColor = hex + '66';
            
            // Dynamic CSS variable override would go here in a robust app
            // For now, we update the primary chart styles
        }
        
        async function updateTheme(colorName) {
            try {
                await fetch(`${API_URL}/settings`, {
                    method: 'POST', headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({theme_color: colorName})
                });
                applyTheme(colorName);
                document.getElementById('settings-modal').classList.add('hidden');
                showToast('THEME_UPDATED', 'success');
                if(currentPortfolioId) refreshPortfolioState(); // re-render charts
            } catch(e) {}
        }

        async function openLeaderboard() {
            try {
                const res = await fetch(`${API_URL}/leaderboard`);
                const data = await res.json();
                const tbody = document.getElementById('leaderboard-tbody');
                tbody.innerHTML = data.leaderboard.map((u, i) => {
                    const color = u.return >= 0 ? 'text-cyber-green' : 'text-cyber-red';
                    const sign = u.return >= 0 ? '+' : '';
                    return `
                        <tr class="border-b border-cyber-cyan/10 hover:bg-white/5">
                            <td class="py-3 pl-2 text-cyber-cyan">#${i+1}</td>
                            <td class="py-3 font-bold">${u.username}</td>
                            <td class="py-3 text-right pr-2 ${color}">${sign}${u.return.toFixed(2)}%</td>
                        </tr>
                    `;
                }).join('');
                document.getElementById('leaderboard-modal').classList.remove('hidden');
            } catch(e) { showToast('ERROR_FETCHING_RANKINGS', 'error'); }
        }

        async function fetchAccountHistory() {
            try {
                const res = await fetch(`${API_URL}/account_history`);
                const data = await res.json();
                if(data.dates.length > 0) {
                    document.getElementById('account-history-panel').classList.remove('hidden');
                    const ctx = document.getElementById('accountHistoryChart').getContext('2d');
                    if(charts.history) charts.history.destroy();
                    charts.history = new Chart(ctx, {
                        type: 'line',
                        data: {
                            labels: data.dates,
                            datasets: [{
                                label: 'Net Worth',
                                data: data.values,
                                borderColor: themeMap[userThemeColor],
                                backgroundColor: themeMap[userThemeColor] + '22',
                                fill: true, tension: 0.1, pointRadius: 2
                            }]
                        },
                        options: { ...mainChartOptions }
                    });
                }
            } catch(e) {}
        }
"""

content = content.replace("const neonColors =", js_updates + "\n        const neonColors =")

# Hook checkAuth to apply theme and load history
checkauth_update = """
                    applyTheme(data.theme_color);
                    fetchAccountHistory();
                    
                    // Trigger Boot Animation
"""
content = content.replace("// Trigger Boot Animation", checkauth_update)

# Handle UPGRADE_REQUIRED errors in trade and watchlist
content = content.replace("showToast(e.message, 'error');", """if(e.message === 'UPGRADE_REQUIRED' || e.message.includes('UPGRADE')) {
                    document.getElementById('upgrade-message').textContent = e.message;
                    document.getElementById('upgrade-modal').classList.remove('hidden');
                } else { showToast(e.message, 'error'); }""")

# Also in executeTrade and addToWatchlist
content = content.replace("if(!res.ok) throw new Error(data.error);", "if(!res.ok) { const err = new Error(data.message || data.error); err.type = data.error; throw err; }")

with open('/Users/saithejas/Desktop/FINAL_PROJECT_MCA/index.html', 'w') as f:
    f.write(content)
print("index.html patched")
