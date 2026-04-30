# 1. Перейти у папку проєкту
cd D:\Projects\ukraine-zones

# 2. Git init
git init
git add .
git commit -m "Initial scaffold"

# 3. Створити і активувати venv
python -m venv .venv
# Якщо PowerShell скаже "running scripts is disabled", виконайте ОДИН раз:
#   Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1

# 4. Поставити залежності
pip install --upgrade pip
pip install -r preprocessing\requirements.txt

# 5. Завантажити OSM extract (~700 MB)
New-Item -ItemType Directory -Path data\raw -Force | Out-Null
Invoke-WebRequest `
  -Uri https://download.geofabrik.de/europe/ukraine-latest.osm.pbf `
  -OutFile data\raw\ukraine-latest.osm.pbf

# 6. Прогнати препроцесинг
cd preprocessing
python 01_extract_settlements.py
python 02_compute_zones.py
python 03_simplify_export.py
cd ..

# 7. Запустити веб
cd web
python -m http.server 8000
# відкрити у браузері http://localhost:8000