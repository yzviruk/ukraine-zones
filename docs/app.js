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
const RIVER_STYLE = { color: "#1f77c4", weight: 1.8, opacity: 0.85 };
const STREAM_STYLE = { color: "#1f77c4", weight: 1.5, opacity: 0.85 };

let riversLayer = null;
let streamsLayer = null;

async function loadAndToggle(file, geojsonOptions, currentRef, checked) {
  if (checked) {
    if (!currentRef.layer) {
      const res = await fetch(`data/${file}`);
      if (!res.ok) {
        alert(`Не вдалось завантажити ${file}`);
        return;
      }
      const geojson = await res.json();
      currentRef.layer = L.geoJSON(geojson, geojsonOptions);
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
  loadAndToggle("rivers.geojson", { style: RIVER_STYLE }, riversRef, toggleRivers.checked)
);

const toggleStreams = document.getElementById("toggle-streams");
toggleStreams.addEventListener("change", () =>
  loadAndToggle("streams.geojson", { style: STREAM_STYLE }, streamsRef, toggleStreams.checked)
);

// Auto-load rivers since checkbox starts checked.
loadAndToggle("rivers.geojson", { style: RIVER_STYLE }, riversRef, true);

// Soils layers (HWSD v2.0, see preprocessing/05_export_soils.py)
const SOIL_LAYERS = [
  { id: "chernozems", color: "#2b1d12" },
  { id: "podzolized_chernozems", color: "#6b4a2b" },
  { id: "grey_forest", color: "#8a8278" },
];

for (const { id, color } of SOIL_LAYERS) {
  const style = { color, weight: 0, fillColor: color, fillOpacity: 0.45 };
  const ref = { layer: null };
  const toggle = document.getElementById(`toggle-${id.replace(/_/g, "-")}`);
  toggle.addEventListener("change", () =>
    // Soils have no popups: let clicks pass through to the markers below/above.
    loadAndToggle(`${id}.geojson`, { style, interactive: false }, ref, toggle.checked)
  );
}

// Chernyakhiv-culture sites (research/chernyakhiv). One point per village:
// exact site locations are deliberately not published (anti-looting).
// Own pane above all polygons, so soil layers never cover the markers.
map.createPane("sites");
map.getPane("sites").style.zIndex = 450;
const CHERNYAKHIV_COLOR = "#c2410c";
const chernyakhivRef = { layer: null };
const chernyakhivOptions = {
  pointToLayer: (feature, latlng) => {
    const p = feature.properties;
    return L.circleMarker(latlng, {
      pane: "sites",
      radius: 3 + 2 * Math.sqrt(p.settlements + p.burials),
      color: "#7c2d12",
      weight: 1,
      fillColor: CHERNYAKHIV_COLOR,
      fillOpacity: 0.8,
    });
  },
  onEachFeature: (feature, layer) => {
    const p = feature.properties;
    const burials = p.burials ? `, могильників: ${p.burials}` : "";
    layer.bindPopup(
      `<b>${p.village}</b> (${p.district} р-н)<br>` +
        `Поселень: ${p.settlements}${burials}<br>` +
        `№ за каталогом: ${p.catalogue_no}<br>` +
        `<small>Точка — центр села; пам'ятки поруч, у межах ~1–2 км.<br>${p.source}</small>`
    );
  },
};

const toggleChernyakhiv = document.getElementById("toggle-chernyakhiv");
toggleChernyakhiv.addEventListener("change", () =>
  loadAndToggle(
    "chernyakhiv_villages.geojson",
    chernyakhivOptions,
    chernyakhivRef,
    toggleChernyakhiv.checked
  )
);

// Chernyakhiv potential (research/chernyakhiv/05d_map.py): model of landscape
// suitability on 2 km cells. Coarse on purpose, like the sites layer.
const POTENTIAL_COLORS = { high: "#6a3d9a", candidate: "#1b9e77", elevated: "#cab2d6" };
const potentialRef = { layer: null };
const potentialOptions = {
  style: (feature) => {
    const p = feature.properties;
    const color = POTENTIAL_COLORS[p.candidate ? "candidate" : p.class];
    return { color, weight: 0, fillColor: color, fillOpacity: p.class === "high" ? 0.6 : 0.45 };
  },
  onEachFeature: (feature, layer) => {
    const p = feature.properties;
    const note = p.candidate
      ? "Сіл з відомими пам'ятками ближче ~3 км немає: або прогалина, або тут не шукали."
      : "";
    layer.bindPopup(
      `<b>Придатність: ${p.label}</b><br>${note}` +
        `<br><small>Модель за розміром найближчого водотоку; клітинки 2 км. ` +
        `Не означає, що тут є пам'ятки.<br>${p.source}</small>`
    );
  },
};
const togglePotential = document.getElementById("toggle-chernyakhiv-potential");
const potentialLegend = document.getElementById("potential-legend");
togglePotential.addEventListener("change", () => {
  potentialLegend.classList.toggle("hidden", !togglePotential.checked);
  loadAndToggle(
    "chernyakhiv_potential.geojson",
    potentialOptions,
    potentialRef,
    togglePotential.checked
  );
});
