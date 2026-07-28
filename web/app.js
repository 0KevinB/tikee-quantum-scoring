// Tikee — demo estática. Todo corre en el navegador, sin backend.

function fmt(n, d = 4) {
  return n === null || n === undefined ? "—" : Number(n).toFixed(d);
}

function renderTabs() {
  const buttons = document.querySelectorAll(".tab-btn");
  const panels = document.querySelectorAll(".panel");
  buttons.forEach((btn) => {
    btn.addEventListener("click", () => {
      buttons.forEach((b) => b.classList.remove("active"));
      panels.forEach((p) => p.classList.remove("active"));
      btn.classList.add("active");
      document.getElementById(btn.dataset.tab).classList.add("active");
    });
  });
}

function renderResultTable(containerId, rows) {
  const el = document.getElementById(containerId);
  let html = `<thead><tr>
    <th>Brazo</th><th>Selección</th><th>Clasif.</th>
    <th>AUC media</th><th>dp</th><th>min</th><th>max</th><th>#Vars</th><th>Jaccard</th>
  </tr></thead><tbody>`;
  rows.forEach((r) => {
    const cls = r.highlight ? "highlight" : r.winner ? "winner" : "";
    html += `<tr class="${cls}">
      <td><strong>${r.brazo}</strong></td>
      <td>${r.seleccion}</td>
      <td>${r.clasif}</td>
      <td class="num">${fmt(r.aucMean)}</td>
      <td class="num">${fmt(r.aucSd)}</td>
      <td class="num">${fmt(r.aucMin)}</td>
      <td class="num">${fmt(r.aucMax)}</td>
      <td class="num">${r.vars}</td>
      <td class="num">${r.jaccard === null ? "—" : fmt(r.jaccard, 3)}</td>
    </tr>`;
  });
  html += "</tbody>";
  el.innerHTML = html;
}

function renderFidelity() {
  const el = document.getElementById("tabla-fidelidad");
  let html = `<thead><tr><th>Sintetizador</th><th>Converge</th><th>Tiempo</th><th>Quality</th><th>MAE correlación objetivo</th></tr></thead><tbody>`;
  TIKEE_DATA.fidelity.forEach((r) => {
    html += `<tr class="${r.best ? "winner" : ""}">
      <td>${r.synth}</td><td>${r.converge}</td><td class="num">${r.tiempo}</td>
      <td class="num">${fmt(r.quality)}</td><td class="num">${fmt(r.mae)}</td>
    </tr>`;
  });
  html += "</tbody>";
  el.innerHTML = html;
}

function renderFairness() {
  const el = document.getElementById("tabla-equidad");
  let html = `<thead><tr><th>Brazo</th><th>#Vars</th><th>AUC_proxy(sexo)</th><th>AUC_proxy(zona_residencia)</th><th>AUC_proxy(provincia)</th></tr></thead><tbody>`;
  TIKEE_DATA.fairness.forEach((r) => {
    html += `<tr class="${r.highlight ? "highlight" : ""}">
      <td><strong>${r.brazo}</strong></td>
      <td class="num">${r.vars}</td>
      <td class="num">${fmt(r.sexo, 3)}</td>
      <td class="num"><strong>${fmt(r.zona, 3)}</strong></td>
      <td class="num">${fmt(r.provincia, 3)}</td>
    </tr>`;
  });
  html += "</tbody>";
  el.innerHTML = html;
}

function renderFriedman() {
  const a = TIKEE_DATA.friedman.nivelA;
  const b = TIKEE_DATA.friedman.nivelB;
  document.getElementById("friedman-a").innerHTML =
    `Estadístico=${a.stat}, p=${a.p} (significativo). Nemenyi CD=${a.cd}. ${a.conclusion}`;
  document.getElementById("friedman-b").innerHTML =
    `Estadístico=${b.stat}, p=${b.p} (significativo). Nemenyi CD=${b.cd}. <strong>${b.conclusion}</strong>`;
}

function renderChecklist() {
  const el = document.getElementById("checklist");
  el.innerHTML = TIKEE_DATA.checkpoints
    .map(
      (c) => `<li>
        <span class="check-icon">${c.ok ? "✓" : "✗"}</span>
        <div><strong>${c.nombre}</strong><br><span style="color:var(--muted)">${c.valor}</span></div>
      </li>`
    )
    .join("");
}

function renderFigures() {
  const el = document.getElementById("figuras");
  el.innerHTML = TIKEE_DATA.figures
    .map(
      (f) => `<figure class="fig">
        <img src="assets/figures/${f.file}" alt="${f.title}" loading="lazy" />
        <figcaption>${f.title}</figcaption>
      </figure>`
    )
    .join("");
}

// ---- Simulador client-side ----

function sigmoid(x) {
  return 1 / (1 + Math.exp(-x));
}

function computeProba(values) {
  const sim = TIKEE_DATA.simulator;
  let z = sim.intercept;
  const contribs = [];
  sim.features.forEach((f) => {
    const standardized = (values[f.key] - f.mean) / f.std;
    const contribution = standardized * f.coef;
    z += contribution;
    contribs.push({ label: f.label, contribution });
  });
  return { proba: sigmoid(z), contribs };
}

function renderSimulator() {
  const sim = TIKEE_DATA.simulator;
  const fieldsEl = document.getElementById("sim-fields");
  document.getElementById("sim-threshold").textContent = fmt(sim.threshold, 4);
  document.getElementById("sim-note").textContent = sim.note;

  const values = {};
  sim.features.forEach((f) => (values[f.key] = f.default));

  fieldsEl.innerHTML = sim.features
    .map(
      (f) => `<div class="field">
        <label for="f-${f.key}">${f.label} <span class="val" id="v-${f.key}">${f.default}</span></label>
        <input type="range" id="f-${f.key}" min="${f.min}" max="${f.max}" step="${f.step}" value="${f.default}" />
      </div>`
    )
    .join("");

  function update() {
    sim.features.forEach((f) => {
      values[f.key] = parseFloat(document.getElementById(`f-${f.key}`).value);
      document.getElementById(`v-${f.key}`).textContent = values[f.key];
    });
    const { proba, contribs } = computeProba(values);
    const pct = (proba * 100).toFixed(1);
    document.getElementById("sim-proba").textContent = `${pct}%`;
    const decisionEl = document.getElementById("sim-decision");
    if (proba >= sim.threshold) {
      decisionEl.textContent = "Rechazado (por encima del umbral)";
      decisionEl.className = "decision rechazado";
    } else {
      decisionEl.textContent = "Aprobado (por debajo del umbral)";
      decisionEl.className = "decision aprobado";
    }

    const maxAbs = Math.max(...contribs.map((c) => Math.abs(c.contribution)), 0.1);
    const contribEl = document.getElementById("sim-contrib");
    contribEl.innerHTML =
      `<div style="color:var(--muted);margin-bottom:8px">Contribución de cada variable al riesgo (rojo = aumenta, verde = reduce)</div>` +
      contribs
        .slice()
        .sort((a, b) => Math.abs(b.contribution) - Math.abs(a.contribution))
        .map((c) => {
          const widthPct = (Math.abs(c.contribution) / maxAbs) * 50;
          const cls = c.contribution >= 0 ? "pos" : "neg";
          return `<div class="contrib-row">
            <div class="name">${c.label}</div>
            <div class="contrib-bar-track"><div class="contrib-bar ${cls}" style="width:${widthPct}%"></div></div>
          </div>`;
        })
        .join("");
  }

  sim.features.forEach((f) => {
    document.getElementById(`f-${f.key}`).addEventListener("input", update);
  });
  update();
}

document.addEventListener("DOMContentLoaded", () => {
  renderTabs();
  renderChecklist();
  renderResultTable("tabla-nivel-a", TIKEE_DATA.nivelA);
  renderResultTable("tabla-nivel-b", TIKEE_DATA.nivelB);
  renderFriedman();
  renderFidelity();
  renderFairness();
  renderFigures();
  renderSimulator();
});
