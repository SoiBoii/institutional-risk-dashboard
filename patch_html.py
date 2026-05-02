import re

with open('index.html', 'r') as f:
    content = f.read()

# 1. Add min-w-0 min-h-0 to cyber-panels containing charts
content = content.replace('class="cyber-panel p-5 flex flex-col cursor-pointer group"', 'class="cyber-panel p-5 flex flex-col cursor-pointer group min-w-0 min-h-0"')
content = content.replace('class="cyber-panel p-5 cursor-pointer group"', 'class="cyber-panel p-5 cursor-pointer group min-w-0 min-h-0"')
content = content.replace('class="cyber-panel p-3 flex flex-col h-[100px] hover:border-cyber-cyan/50 cursor-crosshair"', 'class="cyber-panel p-3 flex flex-col h-[100px] hover:border-cyber-cyan/50 cursor-crosshair min-w-0 min-h-0"')

# 2. Add class="w-full h-full" to canvases
content = re.sub(r'<canvas id="([^"]+)"( class="[^"]*")?></canvas>', r'<canvas id="\1" class="w-full h-full"></canvas>', content)

# 3. Insert mainChartOptions
main_opts = """
        const mainChartOptions = {
            responsive: true,
            maintainAspectRatio: false,
            interaction: { mode: 'nearest', intersect: false },
            plugins: {
                legend: {
                    display: true,
                    labels: { color: '#e2e8f0', font: { family: "'Space Mono', 'Fira Code', monospace", size: 12 }, usePointStyle: true, boxWidth: 8 }
                },
                tooltip: {
                    backgroundColor: 'rgba(9, 9, 11, 0.95)', titleColor: '#00f3ff', bodyColor: '#f8fafc',
                    borderColor: 'rgba(0, 243, 255, 0.4)', borderWidth: 1, titleFont: { family: "'Space Mono', monospace", size: 13 },
                    bodyFont: { family: "'Space Mono', monospace", size: 12 }, padding: 12, cornerRadius: 4, displayColors: true, boxPadding: 4
                }
            },
            scales: {
                x: { grid: { color: 'rgba(255, 255, 255, 0.05)', drawBorder: false }, ticks: { color: '#94a3b8', font: { family: "'Space Mono', monospace", size: 11 } } },
                y: { grid: { color: 'rgba(255, 255, 255, 0.05)', drawBorder: false }, ticks: { color: '#94a3b8', font: { family: "'Space Mono', monospace", size: 11 } } }
            }
        };
"""

content = content.replace("const neonColors = ['#00f3ff', '#bc13fe', '#39ff14', '#ff003c', '#facc15', '#f472b6'];", 
                          "const neonColors = ['#00f3ff', '#bc13fe', '#39ff14', '#ff003c', '#facc15', '#f472b6'];\n" + main_opts)

# 4. Spread mainChartOptions into chart options
# Wealth chart
content = content.replace("options: {\n                    responsive: true, maintainAspectRatio: false,\n                    interaction: { mode: 'index', intersect: false },\n                    scales: { y: { ticks: { callback: v => '$' + v.toLocaleString() } } }\n                }", 
                          "options: {\n                    ...mainChartOptions,\n                    interaction: { mode: 'index', intersect: false },\n                    scales: { ...mainChartOptions.scales, y: { ...mainChartOptions.scales.y, ticks: { ...mainChartOptions.scales.y.ticks, callback: v => '$' + v.toLocaleString() } } }\n                }")

# Backtest chart
content = content.replace("options: { responsive: true, maintainAspectRatio: false, interaction: { mode: 'index', intersect: false }, scales: { y: { ticks: { callback: v => v + '%' } } } }", 
                          "options: { ...mainChartOptions, interaction: { mode: 'index', intersect: false }, scales: { ...mainChartOptions.scales, y: { ...mainChartOptions.scales.y, ticks: { ...mainChartOptions.scales.y.ticks, callback: v => v + '%' } } } }")

# Frontier chart
content = content.replace("options: {\n                    responsive: true, maintainAspectRatio: false,\n                    scales: { x: { title: { display: true, text: 'Volatility %' } }, y: { title: { display: true, text: 'Return %' } } }\n                }", 
                          "options: {\n                    ...mainChartOptions,\n                    scales: { ...mainChartOptions.scales, x: { ...mainChartOptions.scales.x, title: { display: true, text: 'Volatility %' } }, y: { ...mainChartOptions.scales.y, title: { display: true, text: 'Return %' } } }\n                }")

# Underwater chart
content = content.replace("options: { responsive: true, maintainAspectRatio: false, interaction: { mode: 'index', intersect: false }, scales: { y: { ticks: { callback: v => v + '%' } } } }", 
                          "options: { ...mainChartOptions, interaction: { mode: 'index', intersect: false }, scales: { ...mainChartOptions.scales, y: { ...mainChartOptions.scales.y, ticks: { ...mainChartOptions.scales.y.ticks, callback: v => v + '%' } } } }")

# Correlation chart
content = content.replace("options: { responsive: true, maintainAspectRatio: false, interaction: { mode: 'index', intersect: false } }", 
                          "options: { ...mainChartOptions, interaction: { mode: 'index', intersect: false } }")

# Allocation chart
content = content.replace("options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { display: false } } }", 
                          "options: { ...mainChartOptions, plugins: { ...mainChartOptions.plugins, legend: { display: false } }, scales: {} }")

# Factor chart
content = content.replace("options: {\n                    responsive: true, maintainAspectRatio: false,\n                    scales: { r: { grid: { color: 'rgba(0, 243, 255, 0.1)' }, angleLines: { color: 'rgba(0, 243, 255, 0.1)' }, ticks: { display: false } } },\n                    plugins: { legend: { display: false } }\n                }", 
                          "options: {\n                    ...mainChartOptions,\n                    scales: { r: { grid: { color: 'rgba(0, 243, 255, 0.1)' }, angleLines: { color: 'rgba(0, 243, 255, 0.1)' }, ticks: { display: false } } },\n                    plugins: { ...mainChartOptions.plugins, legend: { display: false } }\n                }")

# Sparkline chart
content = content.replace("options: {\n                            responsive: true,\n                            maintainAspectRatio: false,\n                            plugins: { legend: { display: false }, tooltip: { enabled: false } },\n                            scales: { x: { display: false }, y: { display: false } },\n                            layout: { padding: 0 }\n                        }", 
                          "options: {\n                            ...mainChartOptions,\n                            plugins: { ...mainChartOptions.plugins, legend: { display: false }, tooltip: { enabled: false } },\n                            scales: { x: { display: false }, y: { display: false } },\n                            layout: { padding: 0 }\n                        }")

with open('index.html', 'w') as f:
    f.write(content)

print("Patch applied")
