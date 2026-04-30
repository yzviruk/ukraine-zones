# Ukraine Zones Project

## Що робимо
Веб-візуалізація зон України, що знаходяться далі X км від МЕЖІ
найближчого населеного пункту. X налаштовується слайдером 1-10 км.

## Архітектура
- Препроцесинг: Python (pyrosm + Shapely + geopandas) → 10 GeoJSON файлів
- Фронт: статичний HTML + Leaflet.js + noUiSlider
- Без бекенду

## Як визначаються "межі НП"
Стратегія: landuse=residential + fallback.
1. Основа: усі полігони `landuse=residential` з OSM. Це фактична забудована
   територія, найточніший проксі для меж НП.
2. Для нод `place=city/town/village/hamlet`, які НЕ покриті жодним
   residential-полігоном (трапляється для малих сіл), додаємо тип-залежний
   пре-буфер:
       city    -> 2000 m
       town    -> 1000 m
       village ->  300 m
       hamlet  ->  100 m
3. unary_union усього → один MultiPolygon `settlement_areas`.

Stage 2 буферує ЦЮ геометрію, тож відстань міряється від ЕДЖА забудови,
а не від однієї точки в центрі села.

## Дані
- Джерело: https://download.geofabrik.de/europe/ukraine-latest.osm.pbf
- Шари OSM: landuse=residential, place=city/town/village/hamlet, admin_level=2
- Проєкція для обчислень: EPSG:6381 (Ukraine TM, метрична)
- Проєкція для веб: EPSG:4326 (WGS84)

## Команди
- Установка: `cd preprocessing && pip install -r requirements.txt`
- Препроцесинг (повний пайплайн):
  ```
  cd preprocessing
  python 01_extract_settlements.py    # ~2-3 хв
  python 02_compute_zones.py          # ~5-15 хв
  python 03_simplify_export.py        # ~1 хв
  ```
- Запуск веб: `cd web && python -m http.server 8000`
- Тести: `pytest tests/`

## Конвенції
- Python: black + ruff
- JS: vanilla, без транспіляції
- Геометрію зберігаємо у EPSG:4326 у фінальних GeoJSON

## Що НЕ робимо без явного запиту
- Не додаємо React/Vue/Angular
- Не піднімаємо бекенд
- Не змінюємо проєкцію вихідних файлів
- НЕ повертаємось до буферування point-only НП — це систематична помилка

## Структура
```
preprocessing/  ← Python скрипти препроцесингу (3 кроки)
web/            ← статичний фронт; data/ — згенеровані GeoJSON
tests/          ← pytest для геометричних інваріантів
data/raw/       ← ukraine-latest.osm.pbf (gitignore)
data/processed/ ← stage1.gpkg, stage2.gpkg (gitignore)
```

## Перевірки перед коммітом
- `pytest tests/` — зелений (7 тестів)
- `black preprocessing/ tests/` — без diff
- `ruff check preprocessing/ tests/` — без помилок
