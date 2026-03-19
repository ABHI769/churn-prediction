const byId = (id) => document.getElementById(id);

function parseNum(v) {
  if (v === null || v === undefined) return null;
  const s = String(v).trim();
  if (!s) return null;
  const n = Number(s);
  return Number.isFinite(n) ? n : null;
}

function valOrNull(id) {
  const el = byId(id);
  if (!el) return null;
  if (el.tagName === "SELECT") {
    const v = el.value.trim();
    return v ? v : null;
  }
  if (el.type === "number") return parseNum(el.value);
  const v = el.value.trim();
  return v ? v : null;
}

function buildRequest() {
  const customer = {
    customer_id: valOrNull("customer_id") || "C000000",
    age: valOrNull("age"),
    tenure_months: valOrNull("tenure_months"),
    contract_type: valOrNull("contract_type"),
    payment_method: valOrNull("payment_method"),
    paperless_billing: valOrNull("paperless_billing"),
    international_plan: valOrNull("international_plan"),
    avg_monthly_gb: valOrNull("avg_monthly_gb"),
    support_tickets_6m: valOrNull("support_tickets_6m"),
    outages_3m: valOrNull("outages_3m"),
    late_payments_12m: valOrNull("late_payments_12m"),
    app_sessions_30d: valOrNull("app_sessions_30d"),
    add_ons: valOrNull("add_ons"),
    monthly_charges: valOrNull("monthly_charges"),
    total_charges: valOrNull("total_charges"),
  };

  const req = {
    customers: [customer],
    retention_gain: parseNum(byId("retention_gain").value) ?? 120.0,
    retention_cost: parseNum(byId("retention_cost").value) ?? 20.0,
    threshold: parseNum(byId("threshold").value),
    include_shap: !!byId("include_shap").checked,
    include_recommendations: !!byId("include_recommendations").checked,
  };
  return req;
}

function setPreview() {
  const req = buildRequest();
  byId("requestPreview").textContent = JSON.stringify(req, null, 2);
}

function pill(text, tone) {
  const toneMap = {
    good: "bg-emerald-500/15 text-emerald-200 ring-1 ring-emerald-500/30",
    bad: "bg-rose-500/15 text-rose-200 ring-1 ring-rose-500/30",
    neutral: "bg-slate-500/15 text-slate-200 ring-1 ring-white/10",
  };
  const cls = toneMap[tone] || toneMap.neutral;
  return `<span class="inline-flex items-center rounded-full px-2 py-0.5 text-xs ${cls}">${text}</span>`;
}

function renderRecs(recs) {
  const mount = byId("recs");
  mount.innerHTML = "";
  byId("recCount").textContent = `${recs.length} action(s)`;

  for (const r of recs) {
    const card = document.createElement("div");
    card.className = "rounded-2xl bg-slate-900/40 p-4 ring-1 ring-white/10";
    card.innerHTML = `
      <div class="flex items-start justify-between gap-3">
        <div>
          <div class="text-sm font-semibold">${r.action}</div>
          <div class="mt-1 text-sm text-slate-300/80">${r.reason}</div>
        </div>
        <div class="text-right">
          <div class="text-xs text-slate-300/70">est. cost</div>
          <div class="mono text-sm text-slate-200/90">${Number(r.estimated_cost ?? 0).toFixed(2)}</div>
        </div>
      </div>
    `;
    mount.appendChild(card);
  }
}

function renderShapTable(items) {
  const mount = byId("shapTable");
  if (!items || !items.length) {
    mount.innerHTML = `<div class="p-4 text-sm text-slate-300/80">No SHAP data (disabled or unavailable).</div>`;
    return;
  }

  const rows = items
    .map((x) => {
      const dirTone = x.direction === "increases_churn" ? "bad" : "good";
      const dirText = x.direction === "increases_churn" ? "increases churn" : "decreases churn";
      const shap = Number(x.shap ?? 0);
      const shapTone = shap >= 0 ? "bad" : "good";
      return `
        <tr class="border-b border-white/5">
          <td class="px-3 py-2 text-sm font-medium">${x.feature}</td>
          <td class="px-3 py-2">${pill(dirText, dirTone)}</td>
          <td class="px-3 py-2 text-right mono">${pill(shap.toFixed(4), shapTone)}</td>
        </tr>
      `;
    })
    .join("");

  mount.innerHTML = `
    <table class="w-full">
      <thead class="bg-slate-900/60">
        <tr>
          <th class="px-3 py-2 text-left text-xs font-semibold text-slate-300/70">Feature</th>
          <th class="px-3 py-2 text-left text-xs font-semibold text-slate-300/70">Effect</th>
          <th class="px-3 py-2 text-right text-xs font-semibold text-slate-300/70">SHAP</th>
        </tr>
      </thead>
      <tbody class="bg-slate-900/30">${rows}</tbody>
    </table>
  `;
}

function setKpis(result) {
  const p = Number(result.churn_probability ?? NaN);
  const t = Number(result.threshold ?? NaN);
  const will = Number(result.will_churn ?? 0);

  byId("kpiProba").textContent = Number.isFinite(p) ? `${(p * 100).toFixed(1)}%` : "—";
  byId("kpiProbaMeta").textContent = `customer_id: ${result.customer_id ?? "—"}`;

  const decisionText = will === 1 ? "High risk" : "Low risk";
  const decisionTone = will === 1 ? "text-rose-200" : "text-emerald-200";
  byId("kpiDecision").innerHTML = `<span class="${decisionTone}">${decisionText}</span>`;
  byId("kpiDecisionMeta").textContent = `will_churn: ${will}`;

  byId("kpiThreshold").textContent = Number.isFinite(t) ? t.toFixed(3) : "—";
}

async function checkHealth() {
  try {
    const res = await fetch("/health", { method: "GET" });
    const j = await res.json();
    const ok = !!j.model_loaded;
    byId("apiStatus").textContent = ok ? "healthy" : "not loaded";
    byId("apiStatus").className = `text-xs mono ${ok ? "text-emerald-200/80" : "text-rose-200/80"}`;
  } catch (e) {
    byId("apiStatus").textContent = "offline";
    byId("apiStatus").className = "text-xs mono text-rose-200/80";
  }
}

function loadDemo() {
  byId("customer_id").value = "C135790";
  byId("age").value = "41";
  byId("tenure_months").value = "7";
  byId("monthly_charges").value = "92.50";
  byId("total_charges").value = "690.00";
  byId("avg_monthly_gb").value = "78";
  byId("support_tickets_6m").value = "3";
  byId("outages_3m").value = "2";
  byId("late_payments_12m").value = "2";
  byId("app_sessions_30d").value = "4";
  byId("add_ons").value = "2";
  byId("contract_type").value = "month-to-month";
  byId("payment_method").value = "electronic-check";
  byId("paperless_billing").value = "yes";
  byId("international_plan").value = "yes";
  setPreview();
}

async function doPredict() {
  const req = buildRequest();
  setPreview();

  byId("rawResponse").textContent = "Loading…";
  byId("btnPredict").disabled = true;
  byId("btnPredict").classList.add("opacity-60");

  try {
    const res = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(req),
    });
    const j = await res.json();
    byId("rawResponse").textContent = JSON.stringify(j, null, 2);

    const r0 = j?.results?.[0];
    if (!r0) throw new Error("No results returned");

    setKpis(r0);
    renderRecs(r0.recommendations || []);
    renderShapTable(r0.top_shap || []);
  } catch (e) {
    byId("rawResponse").textContent = `Error: ${String(e)}`;
    renderRecs([]);
    renderShapTable([]);
  } finally {
    byId("btnPredict").disabled = false;
    byId("btnPredict").classList.remove("opacity-60");
  }
}

function wireInputs() {
  const ids = [
    "customer_id",
    "age",
    "tenure_months",
    "contract_type",
    "payment_method",
    "paperless_billing",
    "international_plan",
    "avg_monthly_gb",
    "support_tickets_6m",
    "outages_3m",
    "late_payments_12m",
    "app_sessions_30d",
    "add_ons",
    "monthly_charges",
    "total_charges",
    "retention_gain",
    "retention_cost",
    "threshold",
    "include_shap",
    "include_recommendations",
  ];

  for (const id of ids) {
    const el = byId(id);
    if (!el) continue;
    el.addEventListener("input", setPreview);
    el.addEventListener("change", setPreview);
  }
}

window.addEventListener("DOMContentLoaded", async () => {
  wireInputs();
  setPreview();
  await checkHealth();
  setInterval(checkHealth, 6000);

  byId("btnDemo").addEventListener("click", loadDemo);
  byId("btnPredict").addEventListener("click", doPredict);
});

