# Predictive Maintenance System (two datasets)

1. Industrial IoT (canozensoy, 500k rows) - will a machine fail within 7 days?  -> train_industrial.py
2. Microsoft Azure Predictive Maintenance (arnabbiswas1) - will it fail in 24 hours? -> train_azure.py

Files
- train_industrial.py  trains Random Forest, writes model/industrial_model.json (+ .pkl)
- train_azure.py       joins the 5 Azure CSVs (data/azure/), builds rolling features, trains RF -> model/azure_model.json
- website/template.html  the whole frontend (login, sign in, captcha, language, dark/light, themes, dashboard)
- build_site.py        inlines the model into one file: pms-website.html (double-click to open)
- app.py, auth.py      optional FastAPI backend (older, not needed for the demo)

Run: python train_industrial.py ; python build_site.py ; open pms-website.html
Azure: put PdM_*.csv in data/azure/, python train_azure.py data/azure, then in the site's Azure tab click "Import trained Azure model".
Note: login is stored in the browser only (demo); OTP is shown on screen; captcha is a custom math captcha.
