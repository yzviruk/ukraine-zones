# Ukraine Zones

Інтерактивна мапа України, що показує зони, які знаходяться далі X км
від будь-якого населеного пункту. X налаштовується слайдером 1–10 км.

## Швидкий старт

### Передумови
- Python 3.10+
- ~2 GB вільного диску (OSM extract + проміжні файли)
- Інтернет (OSM-тайли у фронтенді)

### 1. Клонувати і поставити залежності
```bash
git clone <your-repo-url> ukraine-zones
cd ukraine-zones
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r preprocessing/requirements.txt
```

### 2. Завантажити OSM-дамп України
```bash
mkdir -p data/raw
curl -L -o data/raw/ukraine-latest.osm.pbf \
  https://download.geofabrik.de/europe/ukraine-latest.osm.pbf
```
Файл ~700 MB. Геофабрик оновлює щодня.

### 3. Прогнати препроцесинг
```bash
cd preprocessing
python 01_extract_settlements.py    # ~1–2 хв
python 02_compute_zones.py          # ~5–15 хв (buffer + difference для 10 рівнів)
python 03_simplify_export.py        # ~1 хв
cd ..
```

Результат: 10 файлів `web/data/zones_{1..10}km.geojson`.

### 4. Запустити веб
```bash
cd web
python -m http.server 8000
```
Відкрити http://localhost:8000

## Тести
```bash
pytest tests/
```

## Структура
- `preprocessing/` — Python-пайплайн (3 кроки)
- `web/` — статичний фронт (Leaflet + noUiSlider)
- `tests/` — інваріанти геометрії
- `data/` — сирі та проміжні файли (не комітимо)

Деталі архітектури та best practices у `DEVELOPMENT_GUIDE.md`
(або в CLAUDE.md для контексту Claude Code).

## Ліцензія
MIT для коду. Дані — © OpenStreetMap contributors (ODbL).
