// Ring-Puzzle-Solver — Frontend
"use strict";

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => document.querySelectorAll(sel);

// --- Ring SVG ---

function renderRingSVG(letters, container) {
  const size = 300;
  const cx = size / 2;
  const cy = size / 2;
  const r = 120;
  const nodeR = 20;

  let svg = `<svg class="ring-svg" viewBox="0 0 ${size} ${size}" xmlns="http://www.w3.org/2000/svg">`;
  svg += `<circle cx="${cx}" cy="${cy}" r="${r}" fill="none" stroke="#30363d" stroke-width="1.5" stroke-dasharray="6 4"/>`;
  svg += `<g font-family="SF Mono, Fira Code, Consolas, monospace" font-size="18" font-weight="700" fill="#e6edf3" text-anchor="middle" dominant-baseline="central">`;

  for (let i = 0; i < letters.length; i++) {
    const angle = (i * 360 / 12 - 90) * Math.PI / 180;
    const x = cx + r * Math.cos(angle);
    const y = cy + r * Math.sin(angle);
    svg += `<circle cx="${x.toFixed(1)}" cy="${y.toFixed(1)}" r="${nodeR}" fill="#1c2333" stroke="#30363d" stroke-width="1.5" data-idx="${i}"/>`;
    svg += `<text x="${x.toFixed(1)}" y="${(y + 1).toFixed(1)}" data-idx="${i}">${letters[i]}</text>`;
  }

  svg += `</g></svg>`;
  container.innerHTML = svg;
}

function highlightRingSVG(container, coveredSet, accentSet) {
  container.querySelectorAll("circle[data-idx]").forEach((el) => {
    const idx = parseInt(el.dataset.idx);
    const letter = el.nextElementSibling?.textContent;
    if (accentSet && accentSet.has(letter)) {
      el.setAttribute("stroke", "#bc8cff");
      el.setAttribute("stroke-width", "2.5");
    } else if (coveredSet && coveredSet.has(letter)) {
      el.setAttribute("stroke", "#3fb950");
      el.setAttribute("stroke-width", "2.5");
    } else {
      el.setAttribute("stroke", "#30363d");
      el.setAttribute("stroke-width", "1.5");
    }
  });
}

// --- Tracking Table ---

function renderTrackingTable(tracking, wordIndex) {
  let html = `<table class="tracking"><thead><tr>`;
  html += `<th>#</th><th>Buchstabe</th><th>Sperrt danach</th><th>N\u00e4chster</th><th></th>`;
  html += `</tr></thead><tbody>`;

  for (const row of tracking) {
    const nextDisplay = row.next || "\u2014";
    let ok = "";
    if (row.next) {
      const isBlocked = row.blocked.includes(row.next);
      ok = isBlocked
        ? `<span class="blocked">\u2718</span>`
        : `<span class="ok">\u2714</span>`;
    }
    html += `<tr>`;
    html += `<td>${row.step}</td>`;
    html += `<td>${row.letter}</td>`;
    html += `<td class="blocked">${row.blocked.join(", ")}</td>`;
    html += `<td>${nextDisplay}</td>`;
    html += `<td>${ok}</td>`;
    html += `</tr>`;
  }

  html += `</tbody></table>`;
  return html;
}

// --- Solution Rendering ---

function renderSolutions(data) {
  const resultsEl = $("#results");
  const ringEl = $("#ring-display");

  if (!data.solutions || data.solutions.length === 0) {
    resultsEl.innerHTML = `
      <div class="solution-box no-solution">
        <div class="label">Kein Ergebnis</div>
        <div class="meta">Keine L\u00f6sung gefunden. Versuche max_words zu erh\u00f6hen oder pr\u00fcfe die Buchstaben.</div>
      </div>`;
    renderRingSVG(data.ring, ringEl);
    return;
  }

  // Stats
  let html = `<div class="stats">`;
  html += `<div class="stat"><span class="value">${data.valid_words_count}</span> g\u00fcltige W\u00f6rter</div>`;
  html += `<div class="stat"><span class="value">${data.solutions.length}</span> L\u00f6sung(en)</div>`;
  html += `<div class="stat"><span class="value">${data.solutions[0].coverage}/12</span> Abdeckung</div>`;
  html += `</div>`;

  // Solutions (max 5)
  const shown = data.solutions.slice(0, 5);
  for (let si = 0; si < shown.length; si++) {
    const sol = shown[si];
    const isPerfect = sol.coverage === 12;
    const boxClass = isPerfect ? "solution-box" : "solution-box no-solution";

    html += `<div class="${boxClass}">`;
    html += `<div class="label">L\u00f6sung #${si + 1} \u2014 ${sol.coverage}/12 Buchstaben</div>`;
    html += `<div class="words">`;
    html += sol.chain.map((w) => `<span>${w}</span>`).join(`<span class="sep">\u2192</span>`);
    html += `</div>`;
    html += `<div class="meta">Abgedeckt: ${sol.covered.join(", ")}</div>`;

    // Tracking details
    if (sol.tracking) {
      html += `<details class="tracking-section"><summary>Tracking anzeigen</summary>`;
      for (let wi = 0; wi < sol.chain.length; wi++) {
        html += `<h3 style="margin-top:0.8rem">Wort ${wi + 1}: ${sol.chain[wi]}</h3>`;
        html += renderTrackingTable(sol.tracking[wi], wi);
        if (wi < sol.chain.length - 1) {
          const curr = sol.chain[wi];
          const next = sol.chain[wi + 1];
          html += `<p style="color:var(--muted);font-size:0.85rem;margin:0.3rem 0">\u27A1 "${curr}" endet mit "${curr[curr.length - 1]}" \u2192 "${next}" beginnt mit "${next[0]}"</p>`;
        }
      }
      html += `</details>`;
    }

    html += `</div>`;
  }

  if (data.solutions.length > 5) {
    html += `<p style="color:var(--muted);font-size:0.85rem;">... und ${data.solutions.length - 5} weitere L\u00f6sungen.</p>`;
  }

  resultsEl.innerHTML = html;

  // Ring SVG with highlighting
  renderRingSVG(data.ring, ringEl);
  if (shown[0]) {
    const coveredSet = new Set(shown[0].covered);
    highlightRingSVG(ringEl, coveredSet, null);
  }
}

// --- API Calls ---

async function solvePuzzle() {
  const letters = $("#letters-input").value.toUpperCase().trim();
  const maxWords = parseInt($("#max-words").value) || 2;

  if (!/^[A-ZÄÖÜ]{12}$/.test(letters)) {
    showError("Bitte genau 12 Buchstaben eingeben (A-Z).");
    return;
  }

  hideError();
  showLoading(true);
  $("#results").innerHTML = "";

  try {
    const res = await fetch("/api/solve", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ letters, max_words: maxWords }),
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || `Server-Fehler (${res.status})`);
    }

    const data = await res.json();
    renderSolutions(data);
  } catch (e) {
    showError(e.message || "Verbindungsfehler");
  } finally {
    showLoading(false);
  }
}

// --- UI Helpers ---

function showLoading(active) {
  const el = $("#loading");
  el.classList.toggle("active", active);
  $("#solve-btn").disabled = active;
}

function showError(msg) {
  const el = $("#error");
  el.textContent = msg;
  el.classList.add("active");
}

function hideError() {
  $("#error").classList.remove("active");
}

// --- Init ---

document.addEventListener("DOMContentLoaded", () => {
  const lettersInput = $("#letters-input");
  const solveBtn = $("#solve-btn");
  const presetSelect = $("#preset");

  // Solve on click
  solveBtn.addEventListener("click", solvePuzzle);

  // Solve on Enter
  lettersInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter") solvePuzzle();
  });

  // Preset selection
  presetSelect.addEventListener("change", () => {
    const val = presetSelect.value;
    if (val) {
      lettersInput.value = val;
    }
  });

  // Initial ring display
  const initial = lettersInput.value.toUpperCase().trim();
  if (initial.length === 12) {
    renderRingSVG(initial.split(""), $("#ring-display"));
  }

  // Update ring on input
  lettersInput.addEventListener("input", () => {
    const val = lettersInput.value.toUpperCase().trim();
    if (val.length === 12) {
      renderRingSVG(val.split(""), $("#ring-display"));
    }
  });
});
