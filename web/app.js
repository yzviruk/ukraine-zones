// Ukraine Zones - frontend
// One pre-computed GeoJSON per integer km in [1..10].
// Slider toggles which layer is visible; layers are cached after first fetch.

const UKRAINE_CENTER = [48.5, 31.5];
const UKRAINE_ZOOM = 6;
const ZONE_STYLE = {
  color: "#b00020",
  weight: 0,
  fillColor: "#b00020",
  fillOpacity: 0.35,
};

const map = L.map("map", { preferCanvas: true }).setView(UKRAINE_CENTER, UKRAINE_ZOOM);

L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
  attribution: "© OpenStreetMap contributors",
  maxZoom: 18,
}).addTo(map);

const cache = {}; // km -> Leaflet GeoJSON layer
let currentLayer = null;
const loadingEl = document.getElementById("loading");

async function loadZone(km) {
  if (cache[km]) return cache[km];
  loadingEl.classList.remove("hidden");
  try {
    const res = await fetch(`data/zones_${km}km.geojson`);
    if (!res.ok) {
      throw new Error(`HTTP ${res.status} for zones_${km}km.geojson`);
    }
    const geojson = await res.json();
    cache[km] = L.geoJSON(geojson, { style: ZONE_STYLE });
    return cache[km];
  } finally {
    loadingEl.classList.add("hidden");
  }
}

async function showZone(km) {
  let layer;
  try {
    layer = await loadZone(km);
  } catch (err) {
    console.error(err);
    alert(
      `Не вдалось завантажити шар ${km} км.\n` +
      `Переконайтесь, що файл web/data/zones_${km}km.geojson існує.\n` +
      `Запустіть препроцесинг: python preprocessing/03_simplify_export.py`
    );
    return;
  }
  if (currentLayer && currentLayer !== layer) {
    map.removeLayer(currentLayer);
  }
  if (!map.hasLayer(layer)) {
    layer.addTo(map);
  }
  currentLayer = layer;
}

const slider = document.getElementById("slider");
noUiSlider.create(slider, {
  start: 5,
  step: 1,
  connect: [true, false],
  range: { min: 1, max: 10 },
});

const labelEl = document.getElementById("km-label");
const labelEl2 = document.getElementById("km-label-2");

slider.noUiSlider.on("update", (vals) => {
  const km = Math.round(parseFloat(vals[0]));
  labelEl.textContent = String(km);
  labelEl2.textContent = String(km);
});

// Only fetch a layer when the user releases the slider, to avoid
// requesting all 10 files while dragging.
slider.noUiSlider.on("change", (vals) => {
  const km = Math.round(parseFloat(vals[0]));
  showZone(km);
});

// Show the initial layer.
showZone(5);

// Rivers layer
const RIVER_STYLE = { color: "#1f77c4", weight: 1.4, opacity: 0.85 };
const STREAM_STYLE = { color: "#1f77c4", weight: 1.0, opacity: 0.7 };

let riversLayer = null;
let streamsLayer = null;

async function loadAndToggle(file, style, currentRef, checked) {
  if (checked) {
    if (!currentRef.layer) {
      const res = await fetch(`data/${file}`);
      if (!res.ok) {
        alert(`Не вдалось завантажити ${file}`);
        return;
      }
      const geojson = await res.json();
      currentRef.layer = L.geoJSON(geojson, { style });
    }
    if (!map.hasLayer(currentRef.layer)) currentRef.layer.addTo(map);
  } else if (currentRef.layer && map.hasLayer(currentRef.layer)) {
    map.removeLayer(currentRef.layer);
  }
}

const riversRef = { layer: null };
const streamsRef = { layer: null };

const toggleRivers = document.getElementById("toggle-rivers");
toggleRivers.addEventListener("change", () =>
  loadAndToggle("rivers.geojson", RIVER_STYLE, riversRef, toggleRivers.checked)
);

const toggleStreams = document.getElementById("toggle-streams");
toggleStreams.addEventListener("change", () =>
  loadAndToggle("streams.geojson", STREAM_STYLE, streamsRef, toggleStreams.checked)
);

// Auto-load rivers since checkbox starts checked.
loadAndToggle("rivers.geojson", RIVER_STYLE, riversRef, true);