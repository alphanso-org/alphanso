"use strict";

const state = {
  config: null,
  bootstrap: null,
  savedConfigId: null,
  savedConfigurations: [],
  platform: { name: "", isMac: false, isWindows: false, modifier: "Ctrl", pathSeparator: "/" },
  isotopeLabels: new Set(),
  result: null,
  elapsed: null,
  running: false,
  chartView: null,
  spectrumMode: "normalized",
};

const $ = (selector, root = document) => root.querySelector(selector);
const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];

const elements = {
  builderView: $("#builder-view"),
  examplesView: $("#examples-view"),
  savedView: $("#saved-view"),
  recentView: $("#recent-view"),
  name: $("#calculation-name"),
  geometryGrid: $("#geometry-grid"),
  geometryFields: $("#geometry-fields"),
  materialDescription: $("#materials-description"),
  runButton: $("#run-button"),
  validationBox: $("#validation-box"),
  validationList: $("#validation-list"),
  results: $("#results-section"),
  yamlFile: $("#yaml-file"),
  examplesGrid: $("#examples-grid"),
  savedGrid: $("#saved-grid"),
  recentGrid: $("#recent-grid"),
  basicParameterFields: $("#basic-parameter-fields"),
  advancedParameterFields: $("#advanced-parameter-fields"),
};

function geometryMeta() {
  return state.bootstrap?.configuration?.calculation_types || {};
}

function commonFields() {
  return state.bootstrap?.configuration?.common_fields || [];
}

function fieldGroupMeta(group) {
  return state.bootstrap?.configuration?.field_groups?.[group] || {};
}

function detectPlatform() {
  const name = navigator.userAgentData?.platform || navigator.platform || "";
  const isMac = /mac|iphone|ipad|ipod/i.test(name);
  const isWindows = /win/i.test(name);
  return {
    name,
    isMac,
    isWindows,
    modifier: isMac ? "Cmd" : "Ctrl",
    pathSeparator: isWindows ? "\\" : "/",
  };
}

function platformPlaceholder(value) {
  if (!state.platform.isWindows || typeof value !== "string") return value;
  return value.replaceAll("/", "\\");
}

function escapeHTML(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function latexHTML(source, displayMode = false) {
  if (!globalThis.katex?.renderToString) {
    return `<code class="latex-source">${escapeHTML(source)}</code>`;
  }
  return katex.renderToString(source, {
    displayMode,
    throwOnError: false,
    strict: "warn",
  });
}

function renderLatex(root = document) {
  $$('[data-latex]', root).forEach(element => {
    if (!globalThis.katex?.render) return;
    katex.render(element.dataset.latex, element, {
      displayMode: element.dataset.latexDisplay === "true",
      throwOnError: false,
      strict: "warn",
    });
  });
}

function labelHTML(label) {
  if (!label || typeof label !== "object" || !label.latex) return escapeHTML(label);
  if (label.text) {
    return `${latexHTML(label.latex)}<span class="metric-label-notation">${escapeHTML(label.notation || "")}</span> ${escapeHTML(label.text)}`;
  }
  return latexHTML(label.latex);
}

function unitHTML(unit) {
  const units = {
    MeV: String.raw`\mathsf{MeV}`,
    cm: String.raw`\mathsf{cm}`,
  };
  return latexHTML(units[unit] || String.raw`\textsf{${unit}}`);
}

function toNumber(value, fallback = 0) {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : fallback;
}

function debounce(fn, delay = 120) {
  let timer;
  return (...args) => {
    clearTimeout(timer);
    timer = setTimeout(() => fn(...args), delay);
  };
}

function uniqueId() {
  if (globalThis.crypto?.randomUUID) return crypto.randomUUID();
  return `${Date.now().toString(36)}-${Math.random().toString(36).slice(2)}`;
}

function geometryVisual(kind) {
  if (kind === "beam") return `<span class="geometry-visual beam-visual" aria-hidden="true"><i></i><b></b></span>`;
  if (kind === "mix") return `<span class="geometry-visual mix-visual" aria-hidden="true"><i></i><i></i><i></i><b></b><b></b></span>`;
  if (kind === "interface") return `<span class="geometry-visual interface-visual" aria-hidden="true"><i></i><b></b></span>`;
  if (kind === "sandwich") return `<span class="geometry-visual sandwich-visual" aria-hidden="true"><i></i><em></em><b></b></span>`;
  return `<span class="geometry-visual" aria-hidden="true"><i></i></span>`;
}

function renderGeometrySelector() {
  elements.geometryGrid.innerHTML = Object.entries(geometryMeta()).map(([key, meta]) => `
    <button class="geometry-card ${key === state.config.calc_type ? "active" : ""}" type="button" data-geometry="${escapeHTML(key)}">
      ${geometryVisual(meta.visual)}
      <span class="geometry-copy"><strong>${escapeHTML(meta.label || key)}</strong><small>${escapeHTML(meta.short_description || meta.description || "")}</small></span>
      <span class="card-check" aria-hidden="true">x</span>
    </button>
  `).join("");
}

function configFieldControl(field) {
  const key = escapeHTML(field.key);
  if (field.kind === "range_points") {
    const values = Array.isArray(state.config[field.key]) ? state.config[field.key] : field.default;
    return `<div class="field-grid three-columns">${field.parts.map((part, index) => `
      <label class="field">
        <span>${escapeHTML(part.label)} ${field.unit && index < 2 ? `<small>${unitHTML(field.unit)}</small>` : ""}</span>
        <input type="number" value="${escapeHTML(values[index])}" data-config-key="${key}" data-config-index="${index}" data-config-kind="${escapeHTML(part.kind)}" ${part.minimum !== undefined ? `min="${escapeHTML(part.minimum)}"` : ""} step="${escapeHTML(part.step ?? "any")}">
      </label>
    `).join("")}</div>`;
  }
  if (field.kind === "boolean") {
    return `<label class="toggle-field compact-toggle">
      <span><strong><code>${key}</code></strong><small>${escapeHTML(field.description || field.label || "")}</small></span>
      <input type="checkbox" data-config-key="${key}" data-config-kind="boolean" ${state.config[field.key] !== false ? "checked" : ""}>
      <i aria-hidden="true"></i>
    </label>`;
  }
  const inputType = ["number", "integer"].includes(field.kind) ? "number" : "text";
  const isDirectory = field.browse === "directory";
  const input = `<input class="${isDirectory ? "path-entry" : ""}" type="${inputType}" value="${escapeHTML(state.config[field.key] ?? field.default ?? "")}" data-config-key="${key}" data-config-kind="${escapeHTML(field.kind)}" ${field.minimum !== undefined ? `min="${escapeHTML(field.minimum)}"` : ""} step="${escapeHTML(field.step ?? "any")}" ${field.placeholder ? `placeholder="${escapeHTML(platformPlaceholder(field.placeholder))}"` : ""} ${isDirectory ? 'autocomplete="off" autocapitalize="off" spellcheck="false"' : ""} title="${escapeHTML(field.label || field.key)}">`;
  return `<label class="field">
    <span><code>${key}</code> ${field.unit ? `<small>${unitHTML(field.unit)}</small>` : ""}</span>
    ${isDirectory ? `<div class="path-input-control">${input}<button type="button" data-browse-directory="${key}" aria-label="Choose folder for ${escapeHTML(field.label || field.key)}">Choose folder</button></div>` : input}
  </label>`;
}

function renderParameterGroups(fields) {
  const groups = new Map();
  fields.forEach(field => {
    if (!groups.has(field.group)) groups.set(field.group, []);
    groups.get(field.group).push(field);
  });
  return [...groups.entries()].map(([group, groupFields]) => {
    const contents = groupFields.map(configFieldControl).join("");
    const hasRange = groupFields.some(field => field.kind === "range_points");
    const columns = group === "Data sources"
      ? "one-column"
      : groupFields.length >= 3 ? "three-columns" : "two-columns";
    const description = fieldGroupMeta(group).description;
    return `<p class="parameter-group-title">${escapeHTML(group)}</p>${description ? `<p class="parameter-group-description">${escapeHTML(description)}</p>` : ""}${hasRange ? contents : `<div class="field-grid ${columns}">${contents}</div>`}`;
  }).join("");
}

function renderParameterFields() {
  const applicable = commonFields().filter(field => !field.applies_to || field.applies_to.includes(state.config.calc_type));
  const standard = applicable.filter(field => !field.advanced);
  const advanced = applicable.filter(field => field.advanced);
  elements.basicParameterFields.innerHTML = renderParameterGroups(standard);
  elements.advancedParameterFields.innerHTML = renderParameterGroups(advanced);
}

function toast(title, detail = "", type = "success") {
  const region = $("#toast-region");
  const item = document.createElement("div");
  item.className = `toast ${type === "error" ? "error" : ""}`;
  item.innerHTML = `<span>${type === "error" ? "!" : "OK"}</span><div><strong>${escapeHTML(title)}</strong>${detail ? `<small>${escapeHTML(detail)}</small>` : ""}</div>`;
  region.append(item);
  window.setTimeout(() => item.remove(), 4600);
}

function materialForRef(ref) {
  if (ref.startsWith("layer:")) {
    const index = Number(ref.split(":")[1]);
    return state.config.intermediate_layers[index].matdef;
  }
  return state.config[ref];
}

function setMaterialForRef(ref, material) {
  if (ref.startsWith("layer:")) {
    const index = Number(ref.split(":")[1]);
    state.config.intermediate_layers[index].matdef = material;
  } else {
    state.config[ref] = material;
  }
}

function materialTotal(material) {
  return Object.values(material || {}).reduce((sum, value) => sum + toNumber(value), 0);
}

function updateMaterialTotal(ref) {
  const card = $(`[data-material="${CSS.escape(ref)}"]`, elements.geometryFields);
  const totalElement = $(".fraction-total", card);
  if (!totalElement) return;
  const total = materialTotal(materialForRef(ref));
  totalElement.classList.toggle("invalid", Math.abs(total - 1) > 1e-5);
  totalElement.innerHTML = `Total <b>${total.toFixed(4)}</b>`;
}

function materialEditor({ title, subtitle, ref, swatch = "target", compact = false }) {
  const material = materialForRef(ref) || {};
  const entries = Object.entries(material);
  const total = materialTotal(material);
  const totalValid = Math.abs(total - 1) <= 1e-5;
  const rows = entries.map(([isotope, fraction]) => `
    <div class="material-row" data-isotope="${escapeHTML(isotope)}">
      <input type="text" list="isotope-options" value="${escapeHTML(isotope)}" aria-label="Isotope" data-action="material-isotope" data-ref="${escapeHTML(ref)}">
      <input type="number" min="0" step="any" value="${escapeHTML(fraction)}" aria-label="Mass fraction" data-action="material-fraction" data-ref="${escapeHTML(ref)}">
      <button class="remove-row" type="button" aria-label="Remove ${escapeHTML(isotope)}" title="Remove isotope" data-action="remove-isotope" data-ref="${escapeHTML(ref)}">x</button>
    </div>
  `).join("");

  return `
    <section class="material-card ${compact ? "compact" : ""}" data-material="${escapeHTML(ref)}">
      <header class="material-card-header">
        <div class="material-title">
          <span class="material-swatch ${escapeHTML(swatch)}"></span>
          <span><strong>${escapeHTML(title)}</strong><small>${escapeHTML(subtitle)}</small></span>
        </div>
        <span class="fraction-total ${totalValid ? "" : "invalid"}">Total <b>${total.toFixed(4)}</b></span>
      </header>
      <div class="material-table">
        <div class="material-table-head"><span>Isotope</span><span>Mass fraction</span><span></span></div>
        <div class="material-rows">${rows}</div>
        <div class="material-actions">
          <button class="text-button" type="button" data-action="add-isotope" data-ref="${escapeHTML(ref)}">+ Add isotope</button>
          <button class="text-button" type="button" data-action="normalize-material" data-ref="${escapeHTML(ref)}">Normalize to 1.0</button>
        </div>
      </div>
    </section>
  `;
}

function beamIntensityEditor() {
  if (state.config.beam_mode === "mono") {
    return `
      <label class="field">
        <span><code>beam_energy</code> <small>${unitHTML("MeV")}</small></span>
        <input type="number" min="0.001" step="0.1" value="${escapeHTML(state.config.beam_energy)}" data-action="beam-energy">
      </label>
    `;
  }
  const rows = state.config.beam_intensities.map(([energy, intensity], index) => `
    <div class="beam-row">
      <input type="number" min="0" step="any" value="${escapeHTML(energy)}" aria-label="Beam energy" data-action="beam-spectrum-energy" data-beam-index="${index}">
      <input type="number" min="0" step="any" value="${escapeHTML(intensity)}" aria-label="Beam intensity" data-action="beam-spectrum-intensity" data-beam-index="${index}">
      <button class="remove-row" type="button" data-action="remove-beam-row" data-beam-index="${index}" aria-label="Remove beam component">x</button>
    </div>
  `).join("");
  return `
    <div class="beam-spectrum-editor">
      <div class="material-table-head"><span>energy ${latexHTML(String.raw`(\mathrm{MeV})`)}</span><span>intensity</span><span></span></div>
      <div class="beam-rows">${rows}</div>
      <button class="text-button" type="button" data-action="add-beam-row">+ Add [energy, intensity]</button>
    </div>
  `;
}

function renderGeometryFields() {
  const geometry = state.config.calc_type;
  elements.materialDescription.textContent = geometryMeta()[geometry]?.description || "Define the calculation materials.";
  let content = "";

  if (geometry === "beam") {
    content = `
      <div class="source-parameters">
        <fieldset class="beam-mode-field">
          <legend>Beam specification</legend>
          <label>
            <input type="radio" name="beam-mode" value="mono" data-action="beam-mode" ${state.config.beam_mode === "mono" ? "checked" : ""}>
            <span><code>beam_energy</code><small>single energy</small></span>
          </label>
          <label>
            <input type="radio" name="beam-mode" value="spectrum" data-action="beam-mode" ${state.config.beam_mode === "spectrum" ? "checked" : ""}>
            <span><code>beam_intensities</code><small>weighted energy spectrum</small></span>
          </label>
        </fieldset>
        ${beamIntensityEditor()}
      </div>
      ${materialEditor({ title: "matdef", subtitle: "Target material mass fractions", ref: "beam_matdef", swatch: "target" })}
    `;
  } else if (geometry === "homogeneous") {
    content = materialEditor({
      title: "matdef",
      subtitle: "Source and target mass fractions",
      ref: "homogeneous_matdef",
      swatch: "source",
    });
  } else if (geometry === "interface") {
    content = `
      <div class="source-parameters one-column">
        <label class="field">
          <span><code>source_density</code> <small>${latexHTML(String.raw`\mathrm{g\,cm^{-3}}`)}</small></span>
          <input type="number" min="0.000001" step="any" value="${escapeHTML(state.config.source_density)}" data-action="source-density">
        </label>
      </div>
      ${materialEditor({ title: "source_matdef", subtitle: "Alpha-emitting Region A", ref: "source_matdef", swatch: "source" })}
      ${materialEditor({ title: "target_matdef", subtitle: "Adjacent target Region B", ref: "target_matdef", swatch: "target" })}
    `;
  } else {
    const layers = state.config.intermediate_layers.map((layer, index) => `
      <section class="layer-card">
        <header class="layer-header">
          <strong>intermediate_layers[${index}]</strong>
          <button type="button" data-action="remove-layer" data-layer="${index}">Remove layer</button>
        </header>
        <div class="layer-fields">
          <label class="field">
            <span><code>density</code> <small>${latexHTML(String.raw`\mathrm{g\,cm^{-3}}`)}</small></span>
            <input type="number" min="0.000001" step="any" value="${escapeHTML(layer.density)}" data-action="layer-density" data-layer="${index}">
          </label>
          <label class="field">
            <span><code>thickness</code> <small>${unitHTML("cm")}</small></span>
            <input type="number" min="0.000000001" step="any" value="${escapeHTML(layer.thickness)}" data-action="layer-thickness" data-layer="${index}">
          </label>
        </div>
        ${materialEditor({ title: "matdef", subtitle: `Mass fractions for layer ${index + 1}`, ref: `layer:${index}`, swatch: "layer", compact: true })}
      </section>
    `).join("");

    content = `
      <div class="source-parameters one-column">
        <label class="field">
          <span><code>source_density</code> <small>${latexHTML(String.raw`\mathrm{g\,cm^{-3}}`)}</small></span>
          <input type="number" min="0.000001" step="any" value="${escapeHTML(state.config.source_density)}" data-action="source-density">
        </label>
      </div>
      ${materialEditor({ title: "source_matdef", subtitle: "Alpha-emitting Region A", ref: "source_matdef", swatch: "source" })}
      <div class="layers-stack">${layers}</div>
      <button class="add-layer-button" type="button" data-action="add-layer">+ Add intermediate layer</button>
      ${materialEditor({ title: "target_matdef", subtitle: "Final target Region C", ref: "target_matdef", swatch: "target" })}
    `;
  }

  elements.geometryFields.innerHTML = content;
  updateSummary();
}

function relevantMaterials() {
  const geometry = state.config.calc_type;
  if (geometry === "beam") return [state.config.beam_matdef];
  if (geometry === "homogeneous") return [state.config.homogeneous_matdef];
  if (geometry === "interface") return [state.config.source_matdef, state.config.target_matdef];
  return [state.config.source_matdef, ...state.config.intermediate_layers.map(layer => layer.matdef), state.config.target_matdef];
}

function clientValidation() {
  const errors = [];
  if (!state.config.name.trim()) errors.push("Enter a calculation name.");

  relevantMaterials().forEach((material, index) => {
    if (!material || !Object.keys(material).length) {
      errors.push(`Material ${index + 1} needs at least one isotope.`);
      return;
    }
    const total = materialTotal(material);
    if (Math.abs(total - 1) > 1e-5) errors.push(`Material ${index + 1} fractions sum to ${total.toFixed(4)}, not 1.0.`);
    Object.entries(material).forEach(([isotope, fraction]) => {
      if (!isotope.trim()) errors.push(`Material ${index + 1} has an empty isotope.`);
      const recognized = state.isotopeLabels.has(isotope) || /^\d+$/.test(isotope);
      if (state.isotopeLabels.size && !recognized) {
        errors.push(`${isotope} is not a recognized isotope, natural element, or ZAID.`);
      }
      if (!(Number(fraction) > 0)) errors.push(`${isotope || "An isotope"} needs a positive mass fraction.`);
    });
  });

  if (state.config.calc_type === "beam" && state.config.beam_mode === "mono" && !(state.config.beam_energy > 0)) {
    errors.push("beam_energy must be greater than zero.");
  }
  if (state.config.calc_type === "beam" && state.config.beam_mode === "spectrum") {
    if (!state.config.beam_intensities.length) errors.push("beam_intensities needs at least one [energy, intensity] pair.");
    state.config.beam_intensities.forEach(([energy, intensity], index) => {
      if (!(energy > 0)) errors.push(`beam_intensities[${index}] energy must be greater than zero.`);
      if (!(intensity > 0)) errors.push(`beam_intensities[${index}] intensity must be greater than zero.`);
    });
  }
  if (["interface", "sandwich"].includes(state.config.calc_type) && !(state.config.source_density > 0)) errors.push("Source density must be greater than zero.");
  if (state.config.calc_type === "sandwich") {
    if (!state.config.intermediate_layers.length) errors.push("Add at least one intermediate layer.");
    state.config.intermediate_layers.forEach((layer, index) => {
      if (!(layer.density > 0)) errors.push(`Layer ${index + 1} density must be greater than zero.`);
      if (!(layer.thickness > 0)) errors.push(`Layer ${index + 1} thickness must be greater than zero.`);
    });
  }

  commonFields()
    .filter(field => !field.applies_to || field.applies_to.includes(state.config.calc_type))
    .forEach(field => {
      const value = state.config[field.key];
      if (field.kind === "integer" && !(Number.isInteger(value) && (field.minimum === undefined || value >= field.minimum))) {
        errors.push(`${field.key} must be an integer${field.minimum !== undefined ? ` of at least ${field.minimum}` : ""}.`);
      } else if (field.kind === "number" && !(Number.isFinite(value) && (field.minimum === undefined || value >= field.minimum))) {
        errors.push(`${field.key} must be a valid number${field.minimum !== undefined ? ` of at least ${field.minimum}` : ""}.`);
      } else if (field.kind === "range_points") {
        if (!Array.isArray(value) || value.length !== 3 || !Number.isFinite(value[0]) || !Number.isFinite(value[1]) || value[0] === value[1]) {
          errors.push(`${field.key} start and stop values must be distinct numbers.`);
        } else if (!Number.isInteger(value[2]) || value[2] < 2) {
          errors.push(`${field.key} num_points must be an integer of at least 2.`);
        }
      }
      if (field.greater_than && !(value > state.config[field.greater_than])) {
        errors.push(`${field.key} must be greater than ${field.greater_than}.`);
      }
    });
  return [...new Set(errors)];
}

function updateSummary() {
  const name = state.config.name.trim() || "Untitled calculation";
  $("#breadcrumb-title").textContent = name;
  $("#save-button").textContent = state.savedConfigId ? "Update configuration" : "Save configuration";
  $("#config-preview").textContent = toYAML(buildRunConfig());

  const errors = clientValidation();
  const valid = errors.length === 0;
  elements.validationBox.classList.toggle("invalid", !valid);
  elements.validationBox.innerHTML = valid
    ? `<span class="validation-icon" aria-hidden="true">OK</span><div><strong>Ready to calculate</strong><small>All required inputs are complete.</small></div>`
    : `<span class="validation-icon" aria-hidden="true">!</span><div><strong>${errors.length} item${errors.length === 1 ? "" : "s"} need attention</strong><small>Review the fields before running.</small></div>`;
  elements.validationList.hidden = valid;
  elements.validationList.innerHTML = errors.slice(0, 5).map(error => `<li>${escapeHTML(error)}</li>`).join("");
  elements.runButton.disabled = !valid || state.running;
}

function syncStaticControls() {
  elements.name.value = state.config.name;
  $$("[data-geometry]", elements.geometryGrid).forEach(card => card.classList.toggle("active", card.dataset.geometry === state.config.calc_type));
}

function renderAll() {
  renderGeometrySelector();
  renderParameterFields();
  syncStaticControls();
  renderGeometryFields();
  updateSummary();
}

function handleMaterialRename(input) {
  const row = input.closest(".material-row");
  const oldName = row.dataset.isotope;
  const newName = input.value.trim();
  const material = materialForRef(input.dataset.ref);
  if (!newName || newName === oldName) {
    updateSummary();
    return;
  }
  if (Object.hasOwn(material, newName)) {
    input.value = oldName;
    toast("Isotope already present", `${newName} is already in this material.`, "error");
    return;
  }
  const rebuilt = {};
  Object.entries(material).forEach(([key, value]) => { rebuilt[key === oldName ? newName : key] = value; });
  setMaterialForRef(input.dataset.ref, rebuilt);
  row.dataset.isotope = newName;
  updateSummary();
}

elements.geometryGrid.addEventListener("click", event => {
  const card = event.target.closest("[data-geometry]");
  if (!card) return;
  state.config.calc_type = card.dataset.geometry;
  state.result = null;
  elements.results.hidden = true;
  renderAll();
});

elements.name.addEventListener("input", event => {
  state.config.name = event.target.value;
  updateSummary();
});

elements.geometryFields.addEventListener("change", event => {
  const action = event.target.dataset.action;
  if (action === "material-isotope") handleMaterialRename(event.target);
  if (action === "beam-mode") {
    state.config.beam_mode = event.target.value;
    renderGeometryFields();
  }
});

elements.geometryFields.addEventListener("input", debounce(event => {
  const input = event.target;
  const action = input.dataset.action;
  if (action === "material-fraction") {
    const row = input.closest(".material-row");
    const material = materialForRef(input.dataset.ref);
    material[row.dataset.isotope] = toNumber(input.value);
    updateMaterialTotal(input.dataset.ref);
    updateSummary();
  } else if (action === "beam-energy") {
    state.config.beam_energy = toNumber(input.value);
    updateSummary();
  } else if (action === "beam-spectrum-energy") {
    state.config.beam_intensities[Number(input.dataset.beamIndex)][0] = toNumber(input.value);
    updateSummary();
  } else if (action === "beam-spectrum-intensity") {
    state.config.beam_intensities[Number(input.dataset.beamIndex)][1] = toNumber(input.value);
    updateSummary();
  } else if (action === "source-density") {
    state.config.source_density = toNumber(input.value);
    updateSummary();
  } else if (action === "layer-density") {
    state.config.intermediate_layers[Number(input.dataset.layer)].density = toNumber(input.value);
    updateSummary();
  } else if (action === "layer-thickness") {
    state.config.intermediate_layers[Number(input.dataset.layer)].thickness = toNumber(input.value);
    updateSummary();
  }
}, 180));

elements.geometryFields.addEventListener("click", event => {
  const button = event.target.closest("button[data-action]");
  if (!button) return;
  const action = button.dataset.action;
  if (action === "remove-isotope") {
    const row = button.closest(".material-row");
    const material = materialForRef(button.dataset.ref);
    delete material[row.dataset.isotope];
    renderGeometryFields();
  } else if (action === "add-isotope") {
    const material = materialForRef(button.dataset.ref);
    const candidates = ["Be-9", "C-13", "O-17", "Al-27", "Pu-238"];
    const isotope = candidates.find(item => !Object.hasOwn(material, item)) || `Isotope-${Object.keys(material).length + 1}`;
    material[isotope] = 0.1;
    renderGeometryFields();
    const editor = $(`[data-material="${CSS.escape(button.dataset.ref)}"]`, elements.geometryFields);
    const inputs = $$('[data-action="material-isotope"]', editor);
    inputs.at(-1)?.focus();
    inputs.at(-1)?.select();
  } else if (action === "normalize-material") {
    const material = materialForRef(button.dataset.ref);
    const total = materialTotal(material);
    if (total > 0) Object.keys(material).forEach(key => { material[key] = material[key] / total; });
    renderGeometryFields();
  } else if (action === "add-layer") {
    state.config.intermediate_layers.push({ matdef: { "Al-27": 1.0 }, density: 2.7, thickness: 0.0001 });
    renderGeometryFields();
  } else if (action === "remove-layer") {
    state.config.intermediate_layers.splice(Number(button.dataset.layer), 1);
    renderGeometryFields();
  } else if (action === "add-beam-row") {
    state.config.beam_intensities.push([5.0, 1.0]);
    renderGeometryFields();
  } else if (action === "remove-beam-row") {
    state.config.beam_intensities.splice(Number(button.dataset.beamIndex), 1);
    renderGeometryFields();
  }
});

function handleConfigFieldInput(event) {
  const input = event.target.closest("[data-config-key]");
  if (!input) return;
  const key = input.dataset.configKey;
  const kind = input.dataset.configKind;
  let value;
  if (kind === "boolean") value = input.checked;
  else if (kind === "integer") value = Number.parseInt(input.value, 10) || 0;
  else if (kind === "number") value = toNumber(input.value);
  else value = input.value;

  if (input.dataset.configIndex !== undefined) {
    const next = [...state.config[key]];
    next[Number(input.dataset.configIndex)] = value;
    state.config[key] = next;
  } else {
    state.config[key] = value;
  }
  updateSummary();
}

elements.basicParameterFields.addEventListener("input", handleConfigFieldInput);
elements.advancedParameterFields.addEventListener("input", handleConfigFieldInput);

document.addEventListener("click", async event => {
  const button = event.target.closest("[data-browse-directory]");
  if (!button) return;
  event.preventDefault();
  const key = button.dataset.browseDirectory;
  const current = state.config[key] || "";
  button.disabled = true;
  try {
    let selected;
    if (window.pywebview?.api?.choose_directory) {
      selected = await window.pywebview.api.choose_directory(current);
    } else {
      selected = window.prompt(`Directory for ${key}`, current);
    }
    if (typeof selected !== "string" || !selected.trim()) return;
    state.config[key] = selected.trim();
    renderParameterFields();
    updateSummary();
  } catch (error) {
    toast("Could not choose directory", error.message, "error");
  } finally {
    button.disabled = false;
  }
});

function toggleAccordion(event, content) {
  const trigger = event.currentTarget;
  const expanded = trigger.getAttribute("aria-expanded") === "true";
  const nextExpanded = !expanded;
  trigger.setAttribute("aria-expanded", String(nextExpanded));
  content.hidden = !nextExpanded;
  trigger.querySelector("[data-accordion-action]").textContent = nextExpanded ? "Close" : "Open";
}

$("#physics-trigger").addEventListener("click", event => toggleAccordion(event, $("#physics-content")));
$("#advanced-trigger").addEventListener("click", event => toggleAccordion(event, $("#advanced-content")));

function buildRunConfig() {
  const config = {
    name: state.config.name.trim(),
    calc_type: state.config.calc_type,
  };
  if (state.config.calc_type === "beam") {
    config.matdef = structuredClone(state.config.beam_matdef);
    if (state.config.beam_mode === "spectrum") config.beam_intensities = structuredClone(state.config.beam_intensities);
    else config.beam_energy = state.config.beam_energy;
  } else if (state.config.calc_type === "homogeneous") {
    config.matdef = structuredClone(state.config.homogeneous_matdef);
  } else {
    config.source_matdef = structuredClone(state.config.source_matdef);
    config.source_density = state.config.source_density;
    config.target_matdef = structuredClone(state.config.target_matdef);
    if (state.config.calc_type === "sandwich") {
      config.intermediate_layers = structuredClone(state.config.intermediate_layers);
    }
  }

  commonFields()
    .filter(field => !field.applies_to || field.applies_to.includes(state.config.calc_type))
    .forEach(field => {
      let value = structuredClone(state.config[field.key]);
      if (field.kind === "integer") value = Math.trunc(value);
      if (field.kind === "range_points") {
        value = value.map((item, index) => field.parts[index].kind === "integer" ? Math.trunc(item) : item);
      }
      if (field.kind === "string") value = value.trim();
      if (field.omit_when_empty && !value) return;
      config[field.key] = value;
    });
  return config;
}

async function runCalculation() {
  if (state.running || clientValidation().length) return;
  state.running = true;
  elements.runButton.classList.add("loading");
  elements.runButton.disabled = true;
  $$("[data-step]").forEach(step => step.classList.toggle("active", step.dataset.step === "results"));
  $$("[data-step='setup'], [data-step='materials']").forEach(step => step.classList.add("complete"));
  try {
    const response = await fetch("/api/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ config: buildRunConfig() }),
    });
    const payload = await response.json();
    if (!response.ok) {
      const detail = Array.isArray(payload.details) ? payload.details.join(" ") : payload.error;
      throw new Error(detail || "The calculation could not be completed.");
    }
    state.result = payload.result;
    state.elapsed = payload.elapsed_seconds;
    state.chartView = null;
    renderResults();
    saveRecentRun();
    elements.results.hidden = false;
    window.setTimeout(() => elements.results.scrollIntoView({ behavior: "smooth", block: "start" }), 50);
    toast("Calculation complete", `Finished in ${formatDuration(state.elapsed)}.`);
  } catch (error) {
    toast("Calculation failed", error.message, "error");
    $$("[data-step]").forEach(step => step.classList.toggle("active", step.dataset.step === "materials"));
  } finally {
    state.running = false;
    elements.runButton.classList.remove("loading");
    updateSummary();
  }
}

elements.runButton.addEventListener("click", runCalculation);
document.addEventListener("keydown", event => {
  const primaryModifier = state.platform.isMac ? event.metaKey : event.ctrlKey;
  if (!primaryModifier) return;
  if (event.key === "Enter") {
    event.preventDefault();
    runCalculation();
  } else if (event.key.toLowerCase() === "s") {
    event.preventDefault();
    saveConfiguration();
  }
});

function formatDuration(seconds) {
  if (!Number.isFinite(seconds)) return "-";
  if (seconds < 1) return `${Math.round(seconds * 1000)} ms`;
  if (seconds < 60) return `${seconds.toFixed(seconds < 10 ? 2 : 1)} s`;
  return `${Math.floor(seconds / 60)}m ${Math.round(seconds % 60)}s`;
}

function sciParts(value, digits = 3) {
  const number = Number(value);
  if (!Number.isFinite(number)) return { html: "-", plain: "-" };
  if (number === 0) return { html: "0", plain: "0" };
  const exponent = Math.floor(Math.log10(Math.abs(number)));
  if (exponent >= -2 && exponent <= 3) {
    const plain = number.toLocaleString("en-US", { maximumSignificantDigits: digits + 1 });
    return { html: latexHTML(String.raw`\mathsf{${plain}}`), plain };
  }
  const mantissa = number / (10 ** exponent);
  return {
    html: latexHTML(String.raw`\mathsf{${mantissa.toFixed(digits)} \times 10^{${exponent}}}`),
    plain: `${mantissa.toFixed(digits)}e${exponent}`,
  };
}

function yieldUnit() {
  if (state.config.calc_type === "beam") return String.raw`\textsf{neutrons per incident }\mathsf{\alpha}`;
  if (state.config.calc_type === "homogeneous") return String.raw`\mathsf{n\,s^{-1}\,g^{-1}}`;
  return String.raw`\mathsf{n\,s^{-1}\,cm^{-2}}`;
}

function metric(label, value, unit, primary = false, valueKind = "numeric") {
  const formatted = valueKind === "duration"
    ? { html: latexHTML(String.raw`\textsf{${formatDuration(value)}}`) }
    : valueKind === "text"
      ? { html: escapeHTML(value) }
      : sciParts(value);
  return `<article class="metric-card ${primary ? "primary" : ""}"><span class="metric-label">${labelHTML(label)}</span><strong class="metric-value">${formatted.html}</strong><span class="metric-unit">${latexHTML(unit)}</span></article>`;
}

function resultSeries(result, mode = state.spectrumMode) {
  const candidates = [
    { label: "Combined", normalized: result.combined_spectrum, strength: result.combined_yield, color: "#17324d", combined: true },
    { label: { latex: String.raw`\mathsf{\alpha\text{-n}}` }, title: "Alpha-n", normalized: result.an_spectrum, absolute: result.an_spectrum_absolute, strength: result.an_yield, color: "#2b6f97" },
    { label: "Spontaneous fission", normalized: result.sf_spectrum, strength: result.sf_yield, color: "#5c806b" },
    { label: "Delayed neutrons", normalized: result.delayedn_spectrum, strength: result.delayedn_strength, color: "#6c6f91" },
  ];
  return candidates.map(series => {
    let values = series.normalized;
    if (mode === "absolute") {
      values = Array.isArray(series.absolute)
        ? series.absolute
        : Array.isArray(series.normalized) && Number.isFinite(Number(series.strength))
          ? series.normalized.map(value => Number(value) * Number(series.strength))
          : null;
    }
    return { ...series, values };
  }).filter(series => Array.isArray(series.values) && series.values.some(value => Number(value) > 0));
}

function absoluteSpectrumUnit() {
  if (state.config.calc_type === "beam") return String.raw`\mathsf{n\,\alpha^{-1}\,bin^{-1}}`;
  if (state.config.calc_type === "homogeneous") return String.raw`\mathsf{n\,s^{-1}\,g^{-1}\,bin^{-1}}`;
  return String.raw`\mathsf{n\,s^{-1}\,cm^{-2}\,bin^{-1}}`;
}

function spectrumAxisScale(yMax) {
  if (state.spectrumMode !== "absolute" || !Number.isFinite(yMax) || yMax <= 0) {
    return { factor: 1, exponent: 0 };
  }
  const exponent = Math.floor(Math.log10(yMax));
  return { factor: 10 ** exponent, exponent };
}

function spectrumTick(value, factor) {
  const scaled = Number(value) / factor;
  if (!Number.isFinite(scaled)) return "-";
  if (scaled === 0) return "0";
  if (Math.abs(scaled) >= 10) return scaled.toFixed(1).replace(/\.0$/, "");
  return scaled.toFixed(2).replace(/0+$/, "").replace(/\.$/, "");
}

function spectrumCenters(bins, count) {
  return Array.from({ length: count }, (_, index) => (
    Number(bins[index]) + Number(bins[index + 1])
  ) / 2);
}

function fullSpectrumDomain(bins, count) {
  const centers = spectrumCenters(bins, count).filter(Number.isFinite);
  const minimum = Math.min(...centers);
  const maximum = Math.max(...centers);
  if (minimum === maximum) return { min: minimum - 0.5, max: maximum + 0.5 };
  return { min: minimum, max: maximum };
}

function visibleSpectrumDomain(full) {
  const view = state.chartView;
  if (!view || !Number.isFinite(view.min) || !Number.isFinite(view.max) || view.max <= view.min) return full;
  return {
    min: Math.max(full.min, view.min),
    max: Math.min(full.max, view.max),
  };
}

function setSpectrumDomain(minimum, maximum, full, count) {
  const fullSpan = full.max - full.min;
  const minimumSpan = Math.max(fullSpan / 200, (2 * fullSpan) / Math.max(count - 1, 1));
  let span = Math.min(fullSpan, Math.max(minimumSpan, maximum - minimum));
  let min = minimum;
  let max = min + span;
  if (min < full.min) {
    min = full.min;
    max = min + span;
  }
  if (max > full.max) {
    max = full.max;
    min = max - span;
  }
  span = max - min;
  state.chartView = span >= fullSpan * 0.999999 ? null : { min, max };
  renderSpectrumChart(state.result || {});
}

function zoomSpectrum(factor, anchorFraction, full, count) {
  const view = visibleSpectrumDomain(full);
  const span = view.max - view.min;
  const normalizedAnchor = Math.min(1, Math.max(0, anchorFraction));
  const anchor = view.min + span * normalizedAnchor;
  const nextSpan = span * factor;
  setSpectrumDomain(
    anchor - nextSpan * normalizedAnchor,
    anchor + nextSpan * (1 - normalizedAnchor),
    full,
    count,
  );
}

function chartPath(values, centers, dimensions, yMax, xMin, xMax) {
  const { left, top, width, height } = dimensions;
  const xScale = value => left + ((value - xMin) / (xMax - xMin || 1)) * width;
  const yScale = value => top + height - (Number(value) / yMax) * height;
  const points = values.map((value, index) => [xScale(centers[index]), yScale(value)]);
  const line = points.map(([x, y], index) => `${index ? "L" : "M"}${x.toFixed(2)},${y.toFixed(2)}`).join(" ");
  return {
    line,
    area: `${line} L${points.at(-1)[0].toFixed(2)},${(top + height).toFixed(2)} L${points[0][0].toFixed(2)},${(top + height).toFixed(2)} Z`,
  };
}

function renderSpectrumChart(result) {
  const container = $("#spectrum-chart");
  const legend = $("#spectrum-legend");
  const bins = result.neutron_energy_bins || result.spectrum_energy_bins;
  const series = resultSeries(result, state.spectrumMode);
  if (!Array.isArray(bins) || bins.length < 2 || !series.length) {
    legend.innerHTML = "";
    container.innerHTML = `<div class="empty-chart">No non-zero neutron spectrum was returned.</div>`;
    return;
  }

  const usable = series.filter(item => item.values.length === bins.length - 1);
  if (!usable.length) {
    legend.innerHTML = "";
    container.innerHTML = `<div class="empty-chart">Spectrum dimensions are not compatible with the returned energy grid.</div>`;
    return;
  }
  legend.innerHTML = usable.map(item => `<span class="legend-item ${item.combined ? "combined" : ""}"><i style="background:${item.color}"></i>${labelHTML(item.label)}</span>`).join("");

  const dimensions = { left: 57, top: 18, width: 630, height: 245 };
  const centers = spectrumCenters(bins, usable[0].values.length);
  const full = fullSpectrumDomain(bins, usable[0].values.length);
  const view = visibleSpectrumDomain(full);
  const visibleValues = usable.flatMap(item => item.values
    .map((value, index) => ({ value: Number(value), energy: centers[index] }))
    .filter(point => point.energy >= view.min && point.energy <= view.max)
    .map(point => point.value));
  const fallbackValues = usable.flatMap(item => item.values.map(Number));
  const yMax = Math.max(...(visibleValues.length ? visibleValues : fallbackValues), Number.EPSILON) * 1.08;
  const axisScale = spectrumAxisScale(yMax);
  const grid = [];
  for (let index = 0; index <= 5; index += 1) {
    const y = dimensions.top + (dimensions.height / 5) * index;
    const value = yMax * (1 - index / 5);
    grid.push(`<line class="chart-grid-line" x1="${dimensions.left}" y1="${y}" x2="${dimensions.left + dimensions.width}" y2="${y}"></line>`);
    grid.push(`<text class="chart-tick" x="${dimensions.left - 9}" y="${y + 3}" text-anchor="end">${spectrumTick(value, axisScale.factor)}</text>`);
  }
  for (let index = 0; index <= 5; index += 1) {
    const x = dimensions.left + (dimensions.width / 5) * index;
    const value = view.min + ((view.max - view.min) / 5) * index;
    grid.push(`<line class="chart-grid-line" x1="${x}" y1="${dimensions.top}" x2="${x}" y2="${dimensions.top + dimensions.height}"></line>`);
    grid.push(`<text class="chart-tick" x="${x}" y="${dimensions.top + dimensions.height + 18}" text-anchor="middle">${value.toFixed(value < 10 ? 1 : 0)}</text>`);
  }
  const combined = usable.find(item => item.combined);
  const areaSeries = combined || usable[0];
  const areaPath = chartPath(areaSeries.values, centers, dimensions, yMax, view.min, view.max);
  const drawOrder = combined ? [...usable.filter(item => !item.combined), combined] : usable;
  const paths = drawOrder.map(item => {
    const path = chartPath(item.values, centers, dimensions, yMax, view.min, view.max);
    return `<path class="spectrum-line ${item.combined ? "combined" : ""}" d="${path.line}" stroke="${item.color}"><title>${escapeHTML(item.title || item.label)}</title></path>`;
  }).join("");
  const rangeLatex = String.raw`\mathsf{${view.min.toFixed(3)}\mathbin{-}${view.max.toFixed(3)}\,MeV}`;
  const absoluteScale = state.spectrumMode === "absolute"
    ? `<div class="chart-y-scale">${latexHTML(`${axisScale.exponent ? String.raw`\times 10^{${axisScale.exponent}}\;` : ""}${absoluteSpectrumUnit()}`)}</div>`
    : "";

  container.innerHTML = `
    <div class="chart-toolbar" aria-label="Spectrum navigation">
      <div class="chart-toolbar-left">
        <div class="chart-mode-switch" role="group" aria-label="Spectrum ordinate">
          <button type="button" data-spectrum-mode="normalized" aria-pressed="${state.spectrumMode === "normalized"}">Normalized</button>
          <button type="button" data-spectrum-mode="absolute" aria-pressed="${state.spectrumMode === "absolute"}">Absolute</button>
        </div>
        <div class="chart-range">${latexHTML(rangeLatex)}</div>
      </div>
      <div class="chart-tools">
        <button type="button" data-chart-action="zoom-in" title="Zoom in">Zoom in</button>
        <button type="button" data-chart-action="zoom-out" title="Zoom out">Zoom out</button>
        <button type="button" data-chart-action="reset" title="Reset spectrum view" ${state.chartView ? "" : "disabled"}>Reset</button>
      </div>
    </div>
    ${absoluteScale}
    <svg class="spectrum-svg" viewBox="0 0 720 292" role="img" aria-label="Interactive neutron energy spectrum">
      <defs><clipPath id="spectrum-plot-clip"><rect x="${dimensions.left}" y="${dimensions.top}" width="${dimensions.width}" height="${dimensions.height}"></rect></clipPath></defs>
      ${grid.join("")}
      <g clip-path="url(#spectrum-plot-clip)">
        <path class="spectrum-area" d="${areaPath.area}" fill="${areaSeries.color}"></path>
        ${paths}
      </g>
      <text class="chart-axis-label" x="12" y="141" text-anchor="middle" transform="rotate(-90 12 141)">${state.spectrumMode === "absolute" ? "Absolute yield per bin" : "Normalized fraction per bin"}</text>
    </svg>
    <div class="chart-x-label">${latexHTML(String.raw`\mathsf{E_n\;(MeV)}`)}</div>
    <div class="chart-hint">Scroll to zoom. Drag to pan. Double-click to reset.</div>
  `;

  container.querySelectorAll("[data-spectrum-mode]").forEach(button => {
    button.addEventListener("click", () => {
      if (button.dataset.spectrumMode === state.spectrumMode) return;
      state.spectrumMode = button.dataset.spectrumMode;
      renderSpectrumChart(state.result || {});
    });
  });

  container.querySelector("[data-chart-action='zoom-in']").addEventListener("click", () => zoomSpectrum(0.65, 0.5, full, centers.length));
  container.querySelector("[data-chart-action='zoom-out']").addEventListener("click", () => zoomSpectrum(1 / 0.65, 0.5, full, centers.length));
  container.querySelector("[data-chart-action='reset']").addEventListener("click", () => {
    state.chartView = null;
    renderSpectrumChart(state.result || {});
  });

  const svg = container.querySelector(".spectrum-svg");
  svg.addEventListener("wheel", event => {
    event.preventDefault();
    const rectangle = svg.getBoundingClientRect();
    const viewBoxX = ((event.clientX - rectangle.left) / rectangle.width) * 720;
    const anchor = (viewBoxX - dimensions.left) / dimensions.width;
    zoomSpectrum(event.deltaY < 0 ? 0.82 : 1 / 0.82, anchor, full, centers.length);
  }, { passive: false });

  let drag = null;
  svg.addEventListener("pointerdown", event => {
    if (event.button !== 0) return;
    drag = { pointerId: event.pointerId, clientX: event.clientX };
    svg.setPointerCapture(event.pointerId);
    svg.classList.add("panning");
  });
  const finishPan = event => {
    if (!drag || drag.pointerId !== event.pointerId) return;
    const rectangle = svg.getBoundingClientRect();
    const plotWidth = rectangle.width * dimensions.width / 720;
    const span = view.max - view.min;
    const movement = event.clientX - drag.clientX;
    const shift = -(movement / plotWidth) * span;
    drag = null;
    svg.classList.remove("panning");
    if (Math.abs(movement) < 2) return;
    setSpectrumDomain(view.min + shift, view.max + shift, full, centers.length);
  };
  svg.addEventListener("pointerup", finishPan);
  svg.addEventListener("pointercancel", () => {
    drag = null;
    svg.classList.remove("panning");
  });
  svg.addEventListener("dblclick", () => {
    state.chartView = null;
    renderSpectrumChart(state.result || {});
  });
}

function renderGammaLines(result) {
  const card = $("#gamma-card");
  const container = $("#gamma-lines");
  const lines = Array.isArray(result.gamma_lines)
    ? result.gamma_lines.filter(line => Array.isArray(line) && Number(line[1]) > 0).sort((a, b) => Number(b[1]) - Number(a[1])).slice(0, 8)
    : [];
  card.hidden = false;
  if (!lines.length) {
    container.innerHTML = `<div class="empty-chart">No discrete gamma lines were returned for this model.</div>`;
    return;
  }
  const maximum = Math.max(...lines.map(line => Number(line[1])));
  container.innerHTML = `<div class="gamma-lines-list">${lines.map(([energy, intensity]) => `
    <div class="gamma-line-row">
      <span class="gamma-energy">${latexHTML(String.raw`\mathsf{${Number(energy).toFixed(3)}\,MeV}`)}</span>
      <span class="gamma-bar"><i style="width:${Math.max(2, (Number(intensity) / maximum) * 100).toFixed(2)}%"></i></span>
      <span class="gamma-intensity">${sciParts(intensity, 2).html}</span>
    </div>
  `).join("")}</div>`;
}

function renderResults() {
  const result = state.result || {};
  const totalYield = result.combined_yield ?? result.an_yield;
  const averageEnergy = result.average_energy ?? result.average_energy_an;
  const alphaYield = result.an_yield;
  const gammaYield = result.gamma_yield;
  $("#result-subtitle").textContent = `${state.config.name} / ${geometryMeta()[state.config.calc_type]?.label || state.config.calc_type} geometry / ${formatDuration(state.elapsed)}`;

  const cards = [
    metric(result.combined_yield !== undefined ? "Combined neutron yield" : { latex: String.raw`\mathsf{\alpha}`, notation: "-n", text: "neutron yield" }, totalYield, yieldUnit(), true),
    alphaYield !== undefined && result.combined_yield !== undefined
      ? metric({ latex: String.raw`\mathsf{\alpha}`, notation: "-n", text: "contribution" }, alphaYield, yieldUnit())
      : metric("Calculation time", state.elapsed, String.raw`\textsf{wall-clock runtime}`, false, "duration"),
    averageEnergy !== undefined
      ? metric("Average neutron energy", averageEnergy, String.raw`\mathsf{MeV}`)
      : metric("Energy groups", state.config.neutron_energy_bins[2] - 1, String.raw`\textsf{neutron bins}`),
    gammaYield !== undefined
      ? metric("Prompt gamma yield", gammaYield, state.config.calc_type === "beam" ? String.raw`\textsf{gammas per incident }\mathsf{\alpha}` : String.raw`\textsf{source-term basis}`)
      : metric("Gamma production", "Disabled", String.raw`\textsf{not calculated}`, false, "text"),
  ];
  $("#metric-grid").innerHTML = cards.join("");
  renderSpectrumChart(result);
  renderGammaLines(result);
}

function yamlScalar(value) {
  if (value === null) return "null";
  if (typeof value === "boolean" || typeof value === "number") return String(value);
  const text = String(value);
  return /^[A-Za-z0-9_.-]+$/.test(text) && !["true", "false", "null"].includes(text.toLowerCase()) ? text : JSON.stringify(text);
}

function toYAML(value, indent = 0) {
  const pad = " ".repeat(indent);
  if (Array.isArray(value)) {
    return value.map(item => {
      if (Array.isArray(item) && item.every(part => part === null || typeof part !== "object")) {
        return `${pad}- [${item.map(yamlScalar).join(", ")}]`;
      }
      if (item && typeof item === "object") {
        const rendered = toYAML(item, indent + 2).split("\n");
        return `${pad}- ${rendered[0].trimStart()}${rendered.length > 1 ? `\n${rendered.slice(1).join("\n")}` : ""}`;
      }
      return `${pad}- ${yamlScalar(item)}`;
    }).join("\n");
  }
  if (value && typeof value === "object") {
    return Object.entries(value).map(([key, item]) => {
      if (Array.isArray(item) && item.every(part => part === null || typeof part !== "object")) {
        return `${pad}${key}: [${item.map(yamlScalar).join(", ")}]`;
      }
      if (item && typeof item === "object") return `${pad}${key}:\n${toYAML(item, indent + 2)}`;
      return `${pad}${key}: ${yamlScalar(item)}`;
    }).join("\n");
  }
  return `${pad}${yamlScalar(value)}`;
}

function safeFilename(name) {
  return (name || "alphanso-calculation").toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "").slice(0, 60) || "alphanso-calculation";
}

function downloadBlob(content, filename, type) {
  const url = URL.createObjectURL(new Blob([content], { type }));
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  anchor.click();
  URL.revokeObjectURL(url);
}

$("#export-button").addEventListener("click", () => {
  downloadBlob(`${toYAML(buildRunConfig())}\n`, `${safeFilename(state.config.name)}.yaml`, "text/yaml");
  toast("Configuration exported", "Saved as a portable YAML input.");
});

$("#download-results").addEventListener("click", () => {
  if (!state.result) return;
  downloadBlob(`${JSON.stringify({ config: buildRunConfig(), results: state.result }, null, 2)}\n`, `${safeFilename(state.config.name)}-results.json`, "application/json");
});

$("#import-button").addEventListener("click", () => elements.yamlFile.click());
elements.yamlFile.addEventListener("change", async event => {
  const file = event.target.files[0];
  event.target.value = "";
  if (!file) return;
  try {
    const response = await fetch("/api/import", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ yaml: await file.text() }),
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || "The YAML file could not be imported.");
    loadConfig(payload.config);
    toast("Configuration imported", payload.warnings?.length ? "Imported with fields that need review." : "The calculation is ready to review.");
  } catch (error) {
    toast("Import failed", error.message, "error");
  }
});

function loadConfig(config, { savedConfigId = null } = {}) {
  const next = structuredClone(state.bootstrap.workspace_defaults);
  Object.assign(next, structuredClone(config));
  next.name = config.name || "Imported calculation";
  next.calc_type = geometryMeta()[config.calc_type] ? config.calc_type : state.bootstrap.workspace_defaults.calc_type;
  if (config.matdef && config.calc_type === "beam") next.beam_matdef = config.matdef;
  if (config.matdef && config.calc_type === "homogeneous") next.homogeneous_matdef = config.matdef;
  next.source_matdef = config.source_matdef || next.source_matdef;
  next.target_matdef = config.target_matdef || next.target_matdef;
  next.intermediate_layers = Array.isArray(config.intermediate_layers) ? config.intermediate_layers : next.intermediate_layers;
  next.beam_mode = Array.isArray(config.beam_intensities) ? "spectrum" : "mono";
  next.beam_intensities = Array.isArray(config.beam_intensities) ? config.beam_intensities : next.beam_intensities;
  if (!Array.isArray(next.neutron_energy_bins) || next.neutron_energy_bins.length !== 3) {
    next.neutron_energy_bins = structuredClone(state.bootstrap.workspace_defaults.neutron_energy_bins);
  }
  state.config = next;
  state.savedConfigId = savedConfigId;
  state.result = null;
  state.chartView = null;
  elements.results.hidden = true;
  showView("builder");
  renderAll();
}

function resetCalculation() {
  if (state.running) return;
  state.config = structuredClone(state.bootstrap.workspace_defaults);
  state.savedConfigId = null;
  state.result = null;
  state.elapsed = null;
  state.chartView = null;
  elements.results.hidden = true;
  $$("[data-step]").forEach(step => {
    step.classList.remove("complete");
    step.classList.toggle("active", step.dataset.step === "setup");
  });
  showView("builder");
  renderAll();
  toast("New calculation", "Started with the default ALPHANSO reference model.");
}

$("#new-button").addEventListener("click", resetCalculation);
$("#edit-inputs").addEventListener("click", () => window.scrollTo({ top: 0, behavior: "smooth" }));

const LEGACY_SAVED_CONFIG_STORAGE_KEY = "alphanso.gui.configs.v1";

function savedConfigurations() {
  return state.savedConfigurations;
}

async function writeSavedConfigurations(configurations) {
  const response = await fetch("/api/saved-configurations", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ configurations: configurations.slice(0, 50) }),
  });
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.error || "Saved configurations could not be written.");
  state.savedConfigurations = Array.isArray(payload.configurations) ? payload.configurations : [];
}

async function saveConfiguration() {
  if (!state.config) return;
  const name = state.config.name.trim();
  if (!name) {
    toast("Name required", "Enter a calculation name before saving.", "error");
    elements.name.focus();
    return;
  }

  const configurations = structuredClone(savedConfigurations());
  let index = state.savedConfigId
    ? configurations.findIndex(item => item.id === state.savedConfigId)
    : -1;
  if (index < 0) {
    index = configurations.findIndex(item => item.config?.name?.trim().toLowerCase() === name.toLowerCase());
  }

  const entry = {
    id: index >= 0 ? configurations[index].id : uniqueId(),
    updated: new Date().toISOString(),
    config: buildRunConfig(),
  };
  const updatedExisting = index >= 0;
  if (updatedExisting) configurations.splice(index, 1);
  configurations.unshift(entry);

  try {
    await writeSavedConfigurations(configurations);
    state.savedConfigId = entry.id;
    renderSavedConfigurations();
    updateSummary();
    toast(updatedExisting ? "Configuration updated" : "Configuration saved", `${name} is stored on this device.`);
  } catch (error) {
    toast("Could not save configuration", error.message, "error");
  }
}

function renderSavedConfigurations() {
  const configurations = savedConfigurations();
  $("#saved-count").textContent = String(configurations.length);
  $("#saved-summary").textContent = configurations.length
    ? `${configurations.length} saved configuration${configurations.length === 1 ? "" : "s"} | local to this device`
    : "No saved configurations.";
  if (!configurations.length) {
    elements.savedGrid.innerHTML = `<div class="empty-state"><div><strong>No saved configurations</strong>Use Save configuration in the workspace to add one.</div></div>`;
    return;
  }

  elements.savedGrid.innerHTML = configurations.map(item => {
    const config = item.config || {};
    const geometry = geometryMeta()[config.calc_type]?.label || config.calc_type || "Unknown";
    const materials = exampleMaterials(config);
    const points = Array.isArray(config.neutron_energy_bins) ? config.neutron_energy_bins[2] : null;
    const dataOverrides = ["an_xs_data_dir", "stopping_power_data_dir", "decay_data_dir", "gamma_data_dir"]
      .filter(key => config[key]).length;
    return `
      <article class="recent-card saved-card">
        <header><h3>${escapeHTML(config.name || "Untitled calculation")}</h3><time>${new Date(item.updated).toLocaleDateString()}</time></header>
        <dl>
          <div><dt>Geometry</dt><dd>${escapeHTML(geometry)}</dd></div>
          <div><dt>Materials</dt><dd>${materials.map(escapeHTML).join(", ") || "-"}</dd></div>
          <div><dt>Grid</dt><dd>${points ? `${escapeHTML(points)} neutron points` : "release default"}</dd></div>
          <div><dt>Data</dt><dd>${dataOverrides ? `${dataOverrides} custom override${dataOverrides === 1 ? "" : "s"}` : "installed datasets"}</dd></div>
        </dl>
        <div class="saved-actions">
          <button class="button button-dark" type="button" data-load-saved="${escapeHTML(item.id)}">Open</button>
          <button class="button button-quiet button-danger" type="button" data-delete-saved="${escapeHTML(item.id)}">Delete</button>
        </div>
      </article>
    `;
  }).join("");
}

$("#save-button").addEventListener("click", saveConfiguration);

function recentRuns() {
  try {
    const parsed = JSON.parse(localStorage.getItem("alphanso.gui.runs.v1") || "[]");
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return [];
  }
}

function saveRecentRun() {
  const runs = recentRuns();
  const total = state.result.combined_yield ?? state.result.an_yield;
  runs.unshift({
    id: uniqueId(),
    timestamp: new Date().toISOString(),
    config: buildRunConfig(),
    summary: { total, unit: yieldUnit(), elapsed: state.elapsed },
  });
  localStorage.setItem("alphanso.gui.runs.v1", JSON.stringify(runs.slice(0, 8)));
  renderRecentRuns();
}

function renderRecentRuns() {
  const runs = recentRuns();
  $("#recent-count").textContent = String(runs.length);
  if (!runs.length) {
    elements.recentGrid.innerHTML = `<div class="empty-state"><div><strong>No calculations yet</strong>Completed runs will appear here for quick reuse.</div></div>`;
    return;
  }
  elements.recentGrid.innerHTML = runs.map(run => `
    <article class="recent-card">
      <header><h3>${escapeHTML(run.config.name || "Untitled calculation")}</h3><time>${new Date(run.timestamp).toLocaleDateString()}</time></header>
      <dl>
        <div><dt>Geometry</dt><dd>${escapeHTML(geometryMeta()[run.config.calc_type]?.label || run.config.calc_type)}</dd></div>
        <div><dt>Yield</dt><dd>${sciParts(run.summary.total, 2).plain}</dd></div>
        <div><dt>Runtime</dt><dd>${escapeHTML(formatDuration(run.summary.elapsed))}</dd></div>
      </dl>
      <button class="button button-dark" type="button" data-load-run="${escapeHTML(run.id)}">Open configuration</button>
    </article>
  `).join("");
}

function exampleMaterials(config) {
  const materials = [];
  for (const key of ["matdef", "source_matdef", "target_matdef"]) {
    if (config[key] && typeof config[key] === "object") materials.push(...Object.keys(config[key]));
  }
  if (Array.isArray(config.intermediate_layers)) {
    config.intermediate_layers.forEach(layer => materials.push(...Object.keys(layer.matdef || {})));
  }
  return [...new Set(materials)];
}

function renderExamples() {
  const examples = Array.isArray(state.bootstrap?.examples) ? state.bootstrap.examples : [];
  const kinds = examples.reduce((counts, example) => {
    const kind = example.kind || "Example";
    counts[kind] = (counts[kind] || 0) + 1;
    return counts;
  }, {});
  $("#example-count").textContent = String(examples.length);
  $("#example-summary").textContent = examples.length
    ? `${examples.length} configurations | ${Object.entries(kinds).map(([kind, count]) => `${count} ${kind}`).join(" | ")}`
    : "No bundled examples were found.";

  elements.examplesGrid.innerHTML = examples.map(example => {
    const config = example.config || {};
    const geometry = geometryMeta()[config.calc_type]?.label || config.calc_type || "Unknown";
    const materials = exampleMaterials(config);
    return `
      <article class="example-card">
        <header>
          <span class="example-type">${escapeHTML(example.kind || "Example")}</span>
          <span class="geometry-tag">${escapeHTML(geometry)}</span>
        </header>
        <div class="example-card-body">
          <h2>${escapeHTML(config.name || example.id)}</h2>
          <p>${escapeHTML(example.description || "Repository example configuration.")}</p>
          <dl>
            <div><dt>Source</dt><dd><code>${escapeHTML(example.source || "example_usage/")}</code></dd></div>
            <div><dt>Materials</dt><dd>${materials.map(escapeHTML).join(", ") || "-"}</dd></div>
          </dl>
        </div>
        <button class="button button-dark" type="button" data-load-example="${escapeHTML(example.id)}">Load configuration</button>
      </article>
    `;
  }).join("");
}

elements.examplesGrid.addEventListener("click", event => {
  const button = event.target.closest("[data-load-example]");
  if (!button) return;
  const example = state.bootstrap?.examples?.find(item => item.id === button.dataset.loadExample);
  if (!example) return;
  loadConfig(example.config);
  toast("Example loaded", `${example.config.name} is ready to review.`);
});

elements.recentGrid.addEventListener("click", event => {
  const button = event.target.closest("[data-load-run]");
  if (!button) return;
  const run = recentRuns().find(item => item.id === button.dataset.loadRun);
  if (run) loadConfig(run.config);
});

elements.savedGrid.addEventListener("click", async event => {
  const loadButton = event.target.closest("[data-load-saved]");
  if (loadButton) {
    const item = savedConfigurations().find(config => config.id === loadButton.dataset.loadSaved);
    if (item) loadConfig(item.config, { savedConfigId: item.id });
    return;
  }

  const deleteButton = event.target.closest("[data-delete-saved]");
  if (!deleteButton) return;
  const configurations = structuredClone(savedConfigurations());
  const item = configurations.find(config => config.id === deleteButton.dataset.deleteSaved);
  if (!item) return;
  if (!window.confirm(`Delete saved configuration "${item.config?.name || "Untitled calculation"}"?`)) return;
  try {
    await writeSavedConfigurations(configurations.filter(config => config.id !== item.id));
    if (state.savedConfigId === item.id) state.savedConfigId = null;
    renderSavedConfigurations();
    updateSummary();
    toast("Saved configuration deleted", item.config?.name || "Untitled calculation");
  } catch (error) {
    toast("Could not delete configuration", error.message, "error");
  }
});

function showView(view) {
  elements.builderView.hidden = view !== "builder";
  elements.examplesView.hidden = view !== "examples";
  elements.savedView.hidden = view !== "saved";
  elements.recentView.hidden = view !== "recent";
  $$("[data-nav]").forEach(item => item.classList.toggle("active", item.dataset.nav === view));
  if (view === "examples") renderExamples();
  if (view === "saved") renderSavedConfigurations();
  if (view === "recent") renderRecentRuns();
  window.scrollTo({ top: 0, behavior: "smooth" });
}

$$('[data-nav]').forEach(item => item.addEventListener("click", () => showView(item.dataset.nav)));
$(".brand").addEventListener("click", event => { event.preventDefault(); showView("builder"); });
$("#workspace-button").addEventListener("click", () => showView("builder"));

async function bootstrap() {
  try {
    const response = await fetch("/api/bootstrap");
    if (!response.ok) throw new Error("Bootstrap endpoint unavailable.");
    state.bootstrap = await response.json();
    state.savedConfigurations = Array.isArray(state.bootstrap.saved_configurations)
      ? structuredClone(state.bootstrap.saved_configurations)
      : [];
    if (!state.savedConfigurations.length) {
      try {
        const legacy = JSON.parse(localStorage.getItem(LEGACY_SAVED_CONFIG_STORAGE_KEY) || "[]");
        if (Array.isArray(legacy) && legacy.length) {
          await writeSavedConfigurations(legacy);
          localStorage.removeItem(LEGACY_SAVED_CONFIG_STORAGE_KEY);
        }
      } catch {
        // Ignore malformed data from releases that used origin-local storage.
      }
    }
    state.platform = detectPlatform();
    state.config = structuredClone(state.bootstrap.workspace_defaults);
    state.isotopeLabels = new Set(state.bootstrap.isotopes.map(item => item.label));
    const optionFragment = document.createDocumentFragment();
    state.bootstrap.isotopes.forEach(isotope => {
      const option = document.createElement("option");
      option.value = isotope.label;
      option.label = isotope.natural
        ? `Z=${isotope.z} - natural element`
        : `Z=${isotope.z}, A=${isotope.a}${Number(isotope.abundance) > 0 ? " - naturally occurring" : ""}`;
      optionFragment.append(option);
    });
    $("#isotope-options").append(optionFragment);
    const packageInfo = state.bootstrap.package || {};
    const displayName = packageInfo.display_name || packageInfo.name || "ALPHANSO";
    const productName = displayName.replace(/\s+GUI$/i, "");
    const productVersion = String(packageInfo.version || "").replace(/^v/i, "");
    $("#app-version").textContent = productVersion ? `v${productVersion}` : "";
    $(".brand-copy strong").textContent = productName;
    $(".brand").setAttribute("aria-label", `${productName} home`);
    document.title = productName;
    $("#shortcut-modifier").textContent = state.platform.modifier;
    elements.runButton.setAttribute("aria-keyshortcuts", state.platform.isMac ? "Meta+Enter" : "Control+Enter");
    $("#save-button").setAttribute("aria-keyshortcuts", state.platform.isMac ? "Meta+S" : "Control+S");
    $("#save-button").title = `${state.platform.modifier}+S`;
    renderAll();
    renderSavedConfigurations();
    renderRecentRuns();
    renderExamples();
    renderLatex();
  } catch (error) {
    toast("Could not load application data", error.message, "error");
  }
}

bootstrap();
