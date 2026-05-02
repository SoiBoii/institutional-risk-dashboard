import re

with open('/Users/saithejas/Desktop/FINAL_PROJECT_MCA/index.html', 'r') as f:
    content = f.read()

# 1. Add credentials to fetch calls
content = re.sub(r"fetch\(`\$\{API_URL\}([^\`]+)`\)", r"fetch(`${API_URL}\1`, { credentials: 'include' })", content)

content = re.sub(r"fetch\(`\$\{API_URL\}([^\`]+)`,\s*\{", r"fetch(`${API_URL}\1`, {\n                    credentials: 'include',", content)

# 2. Fix the split logic for tickers and weights
content = content.replace("document.getElementById('tickers').value.split(',')", "document.getElementById('tickers').value.split(/[\\s,]+/)")
content = content.replace("document.getElementById('weights').value.split(',')", "document.getElementById('weights').value.split(/[\\s,]+/)")

with open('/Users/saithejas/Desktop/FINAL_PROJECT_MCA/index.html', 'w') as f:
    f.write(content)
print("index.html fetched patched")
