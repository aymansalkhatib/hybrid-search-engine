/* ============================================================
   IR Verification Console — front-end logic
   Talks only to this origin; the service reverse-proxies to each
   container at /api/<service>/<path>. No framework, no build step.
   ============================================================ */

const state = {
  meta: null,          // { dashboard, datasets[], services[] }
  health: {},          // key -> 'up' | 'down' | 'planned'
  catalog: null,       // gateway /catalog response
  dataset: null,       // currently focused dataset id
  browse: { offset: 0, limit: 10 },
  qbrowse: { offset: 0, limit: 10 },  // test-queries browser
  selectedQuery: null, // query_id whose qrels are shown
  polling: {},         // datasetId -> intervalId (job pollers)
  indexOptions: null,  // the focused dataset's index preprocessing options (for Normalize)
  repModel: "tfidf",   // representation model in focus
  repPolling: null,    // representation build job poller
  seModel: "bm25",     // search model
  seMode: "parallel",  // hybrid mode
  seFusion: "rrf",     // parallel fusion method
};

/* ---------- tiny helpers ---------- */
const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];
const fmt = (n) => (n == null ? "—" : Number(n).toLocaleString());
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

async function api(service, path = "", { method = "GET", json = null, query = null } = {}) {
  let url = `/api/${service}/${path}`;
  if (query) {
    const qs = new URLSearchParams(
      Object.fromEntries(Object.entries(query).filter(([, v]) => v != null && v !== ""))
    ).toString();
    if (qs) url += (url.includes("?") ? "&" : "?") + qs;
  }
  const opts = { method, headers: {} };
  if (json != null) { opts.headers["Content-Type"] = "application/json"; opts.body = JSON.stringify(json); }
  try {
    const res = await fetch(url, opts);
    let data = null;
    try { data = await res.json(); } catch { /* non-json */ }
    return { ok: res.ok, status: res.status, data };
  } catch (e) {
    return { ok: false, status: 0, data: { error: { message: String(e) } } };
  }
}

function toast(msg, kind = "info", ttl = 3600) {
  const t = document.createElement("div");
  t.className = `toast ${kind}`;
  t.innerHTML = `<span class="t-bar"></span><span>${esc(msg)}</span>`;
  $("#toasts").appendChild(t);
  setTimeout(() => { t.style.opacity = "0"; t.style.transition = ".3s"; setTimeout(() => t.remove(), 300); }, ttl);
}

function errMsg(r) {
  return r?.data?.error?.message || r?.data?.detail || `HTTP ${r?.status || "?"}`;
}

/* ============================================================
   BOOT
   ============================================================ */
async function boot() {
  startClock();
  wireNav();
  wireActions();

  // _meta is served by the console itself (not a proxied service) — fetch directly.
  state.meta = await (await fetch("/api/_meta")).json();
  state.dataset = state.meta.datasets[0] || null;

  fillDatasetSelectors();
  renderServicesSkeleton();
  await refreshAll();
}

async function refreshAll() {
  await checkHealth();
  await loadCatalog();
  renderOverview();
  renderServices();
}

/* ============================================================
   HEALTH
   ============================================================ */
async function checkHealth() {
  const jobs = state.meta.services.map(async (svc) => {
    if (svc.group === "infra" || !svc.proxied) { state.health[svc.key] = "infra"; return; }
    const r = await api(svc.key, "health");
    state.health[svc.key] = r.ok ? "up" : (svc.group === "planned" ? "planned" : "down");
  });
  await Promise.all(jobs);
  // gateway pill
  const pill = $("#gwPill");
  const up = state.health["gateway"] === "up";
  pill.classList.toggle("up", up);
  pill.classList.toggle("down", !up);
  $(".gw-label", pill).textContent = up ? "Gateway online" : "Gateway offline";
}

/* ============================================================
   CATALOG
   ============================================================ */
async function loadCatalog() {
  const r = await api("gateway", "catalog");
  state.catalog = r.ok ? r.data : { datasets: [] };
  if ($(".view[data-view='datasets']").classList.contains("active")) renderDatasets();
  // resume polling any already-running jobs
  for (const d of state.catalog.datasets || []) {
    for (const j of d.active_jobs || []) startPolling(d.dataset_id, j.type, j.job_id);
  }
}

/* ============================================================
   OVERVIEW  (KPIs + architecture)
   ============================================================ */
function renderOverview() {
  // ---- KPIs ----
  const svcAll = state.meta.services.filter((s) => s.group !== "infra");
  const upCount = svcAll.filter((s) => state.health[s.key] === "up").length;
  const cat = state.catalog?.datasets || [];
  const docsInStore = cat.reduce((a, d) => a + (d.ingest?.ingested_count || 0), 0);
  const indexes = cat.filter((d) => d.index?.built).length;

  const kpis = [
    { label: "Services online", val: `${upCount}<small>/${svcAll.length}</small>`, ico: svgGrid },
    { label: "Datasets configured", val: state.meta.datasets.length, ico: svgDb },
    { label: "Docs in store", val: fmt(docsInStore), ico: svgDoc },
    { label: "Indexes built", val: `${indexes}<small>/${state.meta.datasets.length}</small>`, ico: svgList },
  ];
  $("#kpiRow").innerHTML = kpis.map((k) => `
    <div class="kpi">
      <div class="k-top"><span class="k-label">${k.label}</span><span class="k-ico">${k.ico}</span></div>
      <div class="k-val">${k.val}</div>
    </div>`).join("");

  // ---- pipeline mini (based on focused dataset) ----
  const d = cat.find((x) => x.dataset_id === state.dataset) || cat[0];
  setStep("download", d?.download?.downloaded);
  setStep("ingest", (d?.ingest?.ingested_count || 0) > 0);
  setStep("index", d?.index?.built);
  refreshRepresentStep();  // async — lights the "Represent" step if a model is built

  // ---- architecture ----
  renderArch();
}

function setStep(name, done) {
  const el = $(`.pm-step[data-step='${name}']`);
  if (el) el.classList.toggle("done", !!done);
}

function renderArch() {
  const byTier = (t) => state.meta.services.filter((s) => s.tier === t);
  const node = (s) => {
    const h = state.health[s.key];
    const cls = s.group === "planned" ? "planned" : (h === "up" ? "up" : h === "infra" ? "" : "down");
    return `<div class="node ${cls} ${s.tier === 'gateway' ? 'gateway' : ''}">
      <div class="n-top"><span class="n-dot"></span><span class="n-name">${esc(s.label)}</span><span class="n-port">:${s.port}</span></div>
      <div class="n-role">${esc(s.role)}</div>
    </div>`;
  };
  const conn = `<div class="arch-conn"></div>`;
  const clientNode = `<div class="node up">
      <div class="n-top"><span class="n-dot"></span><span class="n-name">Browser · this console</span></div>
      <div class="n-role">One origin → reverse-proxy → every service (no CORS)</div>
    </div>`;

  const tier = (label, list) => `
    <div class="node-row-label">${label}</div>
    <div class="arch-tier">${list.map(node).join("")}</div>`;

  $("#arch").innerHTML = [
    `<div class="arch-tier">${clientNode}</div>`, conn,
    tier("entry point", byTier("gateway")), conn,
    tier("offline pipeline · built", byTier("pipeline")), conn,
    tier("query path · planned", byTier("query")), conn,
    tier("infrastructure", byTier("infra")),
  ].join("");
}

/* ============================================================
   DATASETS
   ============================================================ */
function renderDatasets() {
  const wrap = $("#datasetCards");
  const cat = state.catalog?.datasets || [];
  if (!cat.length) {
    wrap.innerHTML = `<div class="result-empty">No catalog yet — is the gateway up? Check the Services tab.</div>`;
    return;
  }
  wrap.innerHTML = cat.map((d, i) => datasetCard(d, i === 0)).join("");
  cat.forEach((d) => loadDatasetInfo(d.dataset_id));
}

function datasetCard(d, isPrimary) {
  const id = d.dataset_id;
  const downloaded = d.download?.downloaded;
  const ingested = (d.ingest?.ingested_count || 0) > 0;
  const built = d.index?.built;
  const indexDocs = d.index?.num_docs;
  const ingestedCount = d.ingest?.ingested_count || 0;
  // The index is STALE when it doesn't cover the currently-ingested corpus (e.g. an
  // old/partial build left over from before a re-ingest). The corpus & index are
  // decoupled, so ingest never rebuilds — we surface the mismatch instead.
  const stale = built && indexDocs != null && ingestedCount > 0 && indexDocs !== ingestedCount;
  // Pre-fill the "Index preprocessing" controls from the options the existing index
  // was actually built with (so they reflect THIS dataset's index, not static defaults).
  const ixOpts = built && d.index?.options ? d.index.options : null;
  const ppOn = (k, def) => ((ixOpts && k in ixOpts ? !!ixOpts[k] : def) ? "checked" : "");
  const ppMin = ixOpts && ixOpts.min_token_length != null ? ixOpts.min_token_length : 2;
  const job = (d.active_jobs || [])[0];
  // The single "next action" — the only button shown as primary (clear hierarchy).
  // A stale index makes "rebuild" the next step.
  const nextStep = !downloaded ? "download" : (!ingested ? "ingest" : ((!built || stale) ? "index" : null));

  const step = (name, label, done, meta) => `
    <div class="step ${done ? "done" : ""}">
      <div class="st-ic">${done ? "✓" : ""}</div>
      <div class="st-name">${label}</div>
      <div class="st-meta">${meta || ""}</div>
    </div>`;

  const progress = job ? `
    <div class="dprogress indet" id="prog-${cssId(id)}"><i id="bar-${cssId(id)}"></i></div>
    <div class="dprogress-label" id="lbl-${cssId(id)}">
      <span class="msg">${esc(job.type)} starting…</span>
      <span class="pct"><span class="spinner"></span></span>
    </div>` : "";

  return `
  <div class="dcard" data-dataset="${esc(id)}">
    <div class="dcard-head">
      <div class="dcard-title">${esc(id)}</div>
      <span class="dcard-tag ${isPrimary ? "tag-primary" : "tag-bonus"}">${isPrimary ? "primary" : "bonus"}</span>
    </div>

    <div class="dinfo" id="info-${cssId(id)}">
      <div class="di"><div class="di-v">…</div><div class="di-k">docs</div></div>
      <div class="di"><div class="di-v">…</div><div class="di-k">queries</div></div>
      <div class="di"><div class="di-v">…</div><div class="di-k">qrels</div></div>
      <div class="di"><div class="di-v">…</div><div class="di-k">has qrels</div></div>
    </div>

    <div class="stepper">
      ${step("download", "Download", downloaded, downloaded ? fmt(d.download?.doc_count) + " docs" : "—")}
      ${step("ingest", "Ingest", ingested, ingested ? fmt(d.ingest?.ingested_count) + " in DB" : "—")}
      ${stale
        ? `<div class="step stale"><div class="st-ic">!</div><div class="st-name">Index</div><div class="st-meta">${fmt(indexDocs)} ≠ ${fmt(ingestedCount)}</div></div>`
        : step("index", "Index", built, built ? fmt(d.index?.num_docs) + " indexed" : "—")}
    </div>

    ${progress}

    <div class="opt-row">
      <span class="opt-label">General</span>
      <label class="switch sm"><input type="checkbox" data-opt="force"><span></span>Force (re-do)</label>
      <span class="opt-note">Download &amp; ingest always cover the full corpus.</span>
    </div>
    <div class="opt-row">
      <span class="opt-label">Index preprocessing</span>
      ${ixOpts ? `<span class="opt-note">↻ current index's settings</span>` : ""}
      <label class="switch sm"><input type="checkbox" data-pp="lowercase" ${ppOn("lowercase", true)}><span></span>lowercase</label>
      <label class="switch sm"><input type="checkbox" data-pp="remove_stopwords" ${ppOn("remove_stopwords", true)}><span></span>stopwords</label>
      <label class="switch sm"><input type="checkbox" data-pp="lemmatize" ${ppOn("lemmatize", true)}><span></span>lemmatize</label>
      <label class="switch sm"><input type="checkbox" data-pp="stem" ${ppOn("stem", false)}><span></span>stem</label>
      <label class="opt-field">min len <input type="number" class="input-num" data-pp="min_token_length" value="${ppMin}" min="1" max="10"></label>
    </div>

    <div class="dactions">
      <button class="btn btn-sm ${nextStep === "download" ? "btn-primary" : "btn-ghost"}" data-act="download" data-ds="${esc(id)}" ${job ? "disabled" : ""}>Download</button>
      <button class="btn btn-sm ${nextStep === "ingest" ? "btn-primary" : "btn-ghost"}" data-act="ingest" data-ds="${esc(id)}" ${job || !downloaded ? "disabled" : ""} title="${downloaded ? "" : "Download first"}">${ingested ? "Re-ingest" : "Ingest"}</button>
      <button class="btn btn-sm ${nextStep === "index" ? "btn-primary" : "btn-ghost"}" data-act="index" data-ds="${esc(id)}" ${job || !ingested ? "disabled" : ""} title="${ingested ? "" : "Ingest first"}">${built ? (stale ? "Rebuild (stale)" : "Rebuild index") : "Build index"}</button>
      <button class="btn btn-ghost btn-sm spacer" data-act="info" data-ds="${esc(id)}">Info</button>
      <button class="btn btn-danger btn-sm" data-act="delete" data-ds="${esc(id)}" ${job ? "disabled" : ""}>Delete</button>
    </div>
    ${stale
      ? `<p class="hint" style="margin:13px 0 0;color:var(--warn)">⚠ Stale index: built on ${fmt(indexDocs)} docs but the corpus now has ${fmt(ingestedCount)}. <b>Rebuild index</b> to refresh (or Delete to remove it).</p>`
      : built ? `<p class="hint" style="margin:13px 0 0">Index has ${fmt(d.index?.num_docs)} docs (full corpus). To rebuild with different <b>preprocessing</b>, change the options above and enable <b>Force</b>.</p>` : ""}
  </div>`;
}

const cssId = (s) => s.replace(/[^a-z0-9]/gi, "_");

async function loadDatasetInfo(id) {
  const r = await api("gateway", "datasets/info", { query: { dataset: id } });
  const box = $(`#info-${cssId(id)}`);
  if (!box) return;
  if (!r.ok) { box.innerHTML = `<div class="di" style="grid-column:1/-1"><div class="di-k">info unavailable</div></div>`; return; }
  const d = r.data;
  box.innerHTML = `
    <div class="di"><div class="di-v">${fmt(d.doc_count)}</div><div class="di-k">docs</div></div>
    <div class="di"><div class="di-v">${fmt(d.num_queries)}</div><div class="di-k">queries</div></div>
    <div class="di"><div class="di-v">${fmt(d.num_qrels)}</div><div class="di-k">qrels</div></div>
    <div class="di"><div class="di-v ${d.has_qrels ? "qrels-yes" : "qrels-no"}">${d.has_qrels ? "✓" : "✗"}</div><div class="di-k">has qrels</div></div>`;
}

/* ---- dataset actions ---- */
function cardOpts(id) {
  const card = document.querySelector(`.dcard[data-dataset="${id}"]`);
  const force = !!card?.querySelector('[data-opt="force"]')?.checked;
  // Preprocessing options for the index build (the corpus is tokenized with these).
  const options = {};
  card?.querySelectorAll("[data-pp]").forEach((el) => {
    options[el.dataset.pp] = el.type === "checkbox" ? el.checked : (Number(el.value) || 1);
  });
  return { force, options };
}

async function datasetAction(act, id) {
  if (act === "info") {
    const r = await api("gateway", "datasets/info", { query: { dataset: id } });
    toast(r.ok ? `${id}: ${fmt(r.data.doc_count)} docs · qrels: ${r.data.has_qrels ? "yes" : "no"}` : errMsg(r), r.ok ? "info" : "err");
    return;
  }
  if (act === "delete") {
    if (!confirm(`Delete ALL local data for:\n\n${id}\n\n• corpus files (download cache)\n• raw docs in MongoDB\n• the built index (artifact + cache)\n\nReversible only by re-downloading. Continue?`)) return;
    const r = await api("gateway", `datasets?dataset=${encodeURIComponent(id)}&files=true&docs=true&index=true`, { method: "DELETE" });
    if (r.ok) {
      const idx = r.data.index_deleted === true ? " · index removed" : (r.data.index_deleted === false ? " · no index" : "");
      toast(`Deleted ${id} — ${fmt(r.data.docs_deleted)} docs${r.data.files_deleted ? " · files removed" : ""}${idx}`, "ok", 4500);
    } else {
      toast(errMsg(r), "err", 5000);
    }
    await loadCatalog();
    return;
  }

  // download / ingest / index → start a background job, honoring the card's options.
  // Everything covers the FULL corpus (no limits). Only the index carries the
  // preprocessing options.
  const { force, options } = cardOpts(id);
  const payload = act === "index" ? { dataset: id, force, options } : { dataset: id, force };

  const map = { download: "datasets/download", ingest: "datasets/ingest", index: "datasets/index" };
  const r = await api("gateway", map[act], { method: "POST", json: payload });
  if (!r.ok) { toast(`${act} failed: ${errMsg(r)}`, "err", 6000); return; }
  const job = r.data;
  if (job.state === "skipped") {
    toast(`${id}: ${act} already done — enable “Force (re-do)” to run it again`, "info", 5000);
    await loadCatalog();
    return;
  }
  const note = act === "download" ? "" : " (full corpus)";
  toast(`${id}: ${act} started${note}${force ? " · forced" : ""}`, "info");
  await loadCatalog();
  startPolling(id, job.type, job.job_id);
}

/* ---- job polling ---- */
function jobService(type) { return type === "build" ? "indexing" : "docstore"; }

function startPolling(id, type, jobId) {
  const tk = `${id}:${jobId}`;
  if (state.polling[tk]) return;
  const svc = jobService(type);
  state.polling[tk] = setInterval(async () => {
    const r = await api("gateway", `jobs/${svc}/${jobId}`);
    if (!r.ok) return;
    const j = r.data;
    const prog = $(`#prog-${cssId(id)}`);
    const bar = $(`#bar-${cssId(id)}`);
    const lbl = $(`#lbl-${cssId(id)}`);
    // Determinate only once we have real forward progress; otherwise show an
    // animated (indeterminate) bar — e.g. while ir-datasets pulls the archive.
    const det = j.percent != null && j.percent > 0;
    if (prog) prog.classList.toggle("indet", !det);
    if (bar && det) bar.style.width = `${j.percent}%`;
    if (lbl) {
      const counts = j.total ? `(${fmt(j.processed)}/${fmt(j.total)})` : `(${fmt(j.processed)})`;
      lbl.querySelector(".msg").textContent = `${j.type} · ${j.message || j.state} ${counts}`;
      lbl.querySelector(".pct").innerHTML = det ? `${j.percent}%` : `<span class="spinner"></span>`;
    }
    if (["succeeded", "failed", "skipped"].includes(j.state)) {
      clearInterval(state.polling[tk]); delete state.polling[tk];
      toast(`${id}: ${j.type} ${j.state}${j.error ? " — " + j.error : ""}`, j.state === "failed" ? "err" : "ok", 5000);
      await loadCatalog();
      renderOverview();
    }
  }, 1000);
}

/* ============================================================
   PREPROCESS
   ============================================================ */
async function runPreprocess() {
  const text = $("#ppText").value;
  const options = {
    lowercase: $("#ppLower").checked,
    remove_stopwords: $("#ppStop").checked,
    lemmatize: $("#ppLemma").checked,
    stem: $("#ppStem").checked,
    min_token_length: Number($("#ppMin").value) || 1,
  };
  const out = $("#ppOut");
  out.className = "loading"; out.innerHTML = `<span class="spinner"></span> processing…`;
  const r = await api("preprocessing", "preprocess", { method: "POST", json: { text, options } });
  if (!r.ok) { out.className = "result-empty"; out.textContent = `Error: ${errMsg(r)}`; return; }
  const { tokens, normalized_text } = r.data;
  $("#ppCount").hidden = false; $("#ppCount").textContent = `${tokens.length} tokens`;
  out.className = "";
  out.innerHTML = `
    <div class="norm-box">${esc(normalized_text) || "<span style='color:var(--faint)'>(empty)</span>"}</div>
    <div class="chips">${tokens.map((t) => `<span class="chip">${esc(t)}</span>`).join("") || "<span class='hint'>no tokens survived the filters</span>"}</div>`;
}

/* ============================================================
   DOC STORE
   ============================================================ */
async function fetchDocById() {
  const docId = $("#dsDocId").value.trim();
  const out = $("#dsDocOut");
  if (!docId) { out.className = "result-empty"; out.textContent = "Enter a document id."; return; }
  out.className = "loading"; out.innerHTML = `<span class="spinner"></span> fetching…`;
  const r = await api("docstore", "doc", { query: { dataset: state.dataset, doc_id: docId } });
  if (!r.ok) { out.className = "result-empty"; out.textContent = `Not found: ${errMsg(r)}`; return; }
  out.className = "";
  out.innerHTML = `
    <dl class="kv"><dt>doc_id</dt><dd>${esc(r.data.doc_id)}</dd></dl>
    <div class="norm-box doc-text" style="margin-top:10px">${esc(r.data.text)}</div>`;
}

async function browseDocs(reset = false) {
  if (reset) state.browse.offset = 0;
  const { offset, limit } = state.browse;
  const out = $("#dsBrowse");
  out.className = "loading"; out.innerHTML = `<span class="spinner"></span> loading…`;
  const r = await api("docstore", "docs/list", { query: { dataset: state.dataset, offset, limit } });
  if (!r.ok) { out.className = "result-empty"; out.textContent = `Error: ${errMsg(r)} (ingest the dataset first?)`; return; }
  const d = r.data;
  $("#dsPageInfo").textContent = d.total ? `${fmt(offset + 1)}–${fmt(Math.min(offset + limit, d.total))} of ${fmt(d.total)}` : "0 docs";
  $("#dsPrev").disabled = offset <= 0;
  $("#dsNext").disabled = offset + limit >= d.total;
  if (!d.docs.length) { out.className = "result-empty"; out.textContent = "No documents stored for this dataset yet."; return; }
  out.className = "";
  out.innerHTML = `<table class="tbl">
    <thead><tr><th style="width:60px">seq</th><th style="width:120px">doc_id</th><th>original text</th></tr></thead>
    <tbody>${d.docs.map((x) => `<tr>
      <td class="mono">${x.seq}</td>
      <td class="mono">${esc(x.doc_id)}</td>
      <td class="doc-text">${esc(trunc(x.text, 260))}</td></tr>`).join("")}</tbody></table>`;
}
const trunc = (s, n) => (s && s.length > n ? s.slice(0, n) + " …" : s || "");

/* ---- test queries + qrels ---- */
async function browseQueries(reset = false) {
  if (reset) { state.qbrowse.offset = 0; state.selectedQuery = null; renderQrels(null); }
  const { offset, limit } = state.qbrowse;
  const out = $("#dsQueries");
  out.className = "loading"; out.innerHTML = `<span class="spinner"></span> loading…`;
  const r = await api("docstore", "queries", { query: { dataset: state.dataset, offset, limit } });
  if (!r.ok) { out.className = "result-empty"; out.textContent = `Error: ${errMsg(r)} (ingest the dataset first?)`; return; }
  const d = r.data;
  $("#qPageInfo").textContent = d.total ? `${fmt(offset + 1)}–${fmt(Math.min(offset + limit, d.total))} of ${fmt(d.total)}` : "0 queries";
  $("#qPrev").disabled = offset <= 0;
  $("#qNext").disabled = offset + limit >= d.total;
  if (!d.queries.length) { out.className = "result-empty"; out.textContent = "No queries stored for this dataset yet."; return; }
  out.className = "";
  out.innerHTML = `<table class="tbl">
    <thead><tr><th style="width:90px">query_id</th><th>text</th><th style="width:74px"></th></tr></thead>
    <tbody>${d.queries.map((x) => `<tr class="q-row${x.query_id === state.selectedQuery ? " sel" : ""}" data-qid="${esc(x.query_id)}">
      <td class="mono">${esc(x.query_id)}</td>
      <td>${esc(trunc(x.text, 200))}</td>
      <td><button class="btn btn-ghost btn-sm" data-qid="${esc(x.query_id)}">qrels ›</button></td></tr>`).join("")}</tbody></table>`;
}

async function loadQrelsForQuery(queryId) {
  state.selectedQuery = queryId;
  $$("#dsQueries .q-row").forEach((tr) => tr.classList.toggle("sel", tr.dataset.qid === queryId));
  const out = $("#dsQrels");
  out.className = "loading"; out.innerHTML = `<span class="spinner"></span> loading judgments…`;
  const r = await api("docstore", "qrels", { query: { dataset: state.dataset, query_id: queryId } });
  if (!r.ok) { out.className = "result-empty"; out.textContent = `Error: ${errMsg(r)}`; return; }
  renderQrels(r.data);
}

function renderQrels(d) {
  const out = $("#dsQrels");
  if (!d) { out.className = "result-empty"; out.textContent = "Pick a query on the left to see its judged documents."; return; }
  if (!d.judgments.length) { out.className = "result-empty"; out.textContent = `No judgments stored for query ${esc(d.query_id)}.`; return; }
  out.className = "";
  out.innerHTML = `<div class="hint">Judged docs for query <b class="mono">${esc(d.query_id)}</b> — ${fmt(d.judgments.length)} judgment(s).</div>
    <table class="tbl">
      <thead><tr><th>doc_id</th><th style="width:110px">relevance</th></tr></thead>
      <tbody>${d.judgments.map((j) => `<tr>
        <td class="mono">${esc(j.doc_id)}</td>
        <td>${esc(j.relevance)}</td></tr>`).join("")}</tbody></table>`;
}

/* ============================================================
   INDEX EXPLORER
   ============================================================ */
async function loadIndexStats() {
  const grid = $("#ixStats");
  grid.innerHTML = `<div class="loading" style="grid-column:1/-1"><span class="spinner"></span> loading index stats…</div>`;
  const r = await api("indexing", "stats", { query: { dataset: state.dataset } });
  if (!r.ok) { state.indexOptions = null; grid.innerHTML = `<div class="result-empty" style="grid-column:1/-1">No index for <b>${esc(state.dataset)}</b>. Build it from the Datasets tab. (${esc(errMsg(r))})</div>`; return; }
  const s = r.data;
  state.indexOptions = s.options;  // so Term lookup → Normalize uses the SAME options
  const cards = [
    ["Documents", fmt(s.num_docs), s.version],
    ["Vocabulary", fmt(s.vocab_size), "unique terms"],
    ["Postings", fmt(s.num_postings), "(term,doc) pairs"],
    ["Total tokens", fmt(s.total_tokens), "after preprocessing"],
    ["avgdl", (s.avg_doc_length || 0).toFixed(2), "BM25 average length"],
    ["Built at", (s.built_at || "").replace("T", " ").slice(0, 19), s.cached ? "cached" : "fresh"],
  ];
  grid.innerHTML = cards.map((c) => `
    <div class="stat"><div class="s-label">${c[0]}</div><div class="s-val">${c[1]}</div><div class="s-sub">${esc(c[2])}</div></div>`).join("");
}

async function normalizeTerm() {
  const raw = $("#ixTerm").value.trim();
  if (!raw) return;
  // A term only matches the index if normalized with the SAME options the index used.
  const opts = state.indexOptions;
  if (!opts) { toast("Load index stats first — no index options available for this dataset", "err"); return; }
  const r = await api("preprocessing", "preprocess", { method: "POST", json: { text: raw, options: opts } });
  if (r.ok && r.data.tokens.length) {
    $("#ixTerm").value = r.data.tokens[0];
    toast(`Normalized with the index's options → "${r.data.tokens[0]}"`, "info");
  } else {
    toast("Nothing left after normalization — this term is a stopword / too short for this index", "err", 5000);
  }
}

async function lookupTerm() {
  const term = $("#ixTerm").value.trim();
  const out = $("#ixTermOut");
  if (!term) { out.className = "result-empty"; out.textContent = "Enter a term."; return; }
  out.className = "loading"; out.innerHTML = `<span class="spinner"></span> looking up…`;
  const r = await api("indexing", "postings", { query: { dataset: state.dataset, term, limit: 50 } });
  if (!r.ok) { out.className = "result-empty"; out.textContent = `Error: ${errMsg(r)}`; return; }
  const d = r.data;
  out.className = "";
  out.innerHTML = `
    <dl class="kv">
      <dt>term</dt><dd>${esc(d.term)}</dd>
      <dt>df (docs)</dt><dd>${fmt(d.df)}</dd>
      <dt>cf (total tf)</dt><dd>${fmt(d.cf)}</dd>
    </dl>
    ${d.postings.length ? `<table class="tbl" style="margin-top:12px">
      <thead><tr><th>doc_id</th><th style="width:80px">tf</th></tr></thead>
      <tbody>${d.postings.map((p) => `<tr><td class="mono">${esc(p.doc_id)}</td><td class="mono">${p.tf}</td></tr>`).join("")}</tbody>
    </table>` : `<p class="hint">No postings — term not in this index.</p>`}`;
}

async function lookupDoc() {
  const docId = $("#ixDocId").value.trim();
  const out = $("#ixDocOut");
  if (!docId) { out.className = "result-empty"; out.textContent = "Enter a document id."; return; }
  out.className = "loading"; out.innerHTML = `<span class="spinner"></span> looking up…`;
  const r = await api("indexing", "doc", { query: { dataset: state.dataset, doc_id: docId } });
  if (!r.ok) { out.className = "result-empty"; out.textContent = `Error: ${errMsg(r)}`; return; }
  out.className = "";
  out.innerHTML = `<dl class="kv"><dt>doc_id</dt><dd>${esc(r.data.doc_id)}</dd><dt>length (tokens)</dt><dd>${fmt(r.data.length)}</dd></dl>`;
}

/* ============================================================
   REPRESENTATIONS  (build · track · inspect the algorithm)
   ============================================================ */
function repPreprocessOptions() {
  const opts = {};
  $$("[data-rpp]").forEach((el) => {
    opts[el.dataset.rpp] = el.type === "checkbox" ? el.checked : (Number(el.value) || 1);
  });
  return opts;
}
function repParams() {
  return {
    min_df: Number($("#repMinDf").value) || 1,
    max_df: Number($("#repMaxDf").value) || 1.0,
    sublinear_tf: $("#repSublinear").checked,
  };
}
function applyRepControls(s) {
  if (s.params) {
    if (s.params.min_df != null) $("#repMinDf").value = s.params.min_df;
    if (s.params.max_df != null) $("#repMaxDf").value = s.params.max_df;
    if (s.params.sublinear_tf != null) $("#repSublinear").checked = !!s.params.sublinear_tf;
  }
  if (s.options) $$("[data-rpp]").forEach((el) => {
    const k = el.dataset.rpp;
    if (k in s.options) { if (el.type === "checkbox") el.checked = !!s.options[k]; else el.value = s.options[k]; }
  });
}

async function loadRepresentation() {
  const model = state.repModel;
  // Stop any poller from the previous dataset/model up front, so switching focus
  // never leaves a stale progress bar from another dataset's build on screen.
  resetRepProgress();
  $("#repModelBadge").textContent = model;
  repSyncControls();
  const grid = $("#repStats");
  grid.innerHTML = `<div class="loading" style="grid-column:1/-1"><span class="spinner"></span> loading representation status…</div>`;
  const reqDataset = state.dataset;
  const r = await api("representation", "status", { query: { dataset: reqDataset, model } });
  // The user may have switched focus again while this was in flight — drop a stale response.
  if (reqDataset !== state.dataset || model !== state.repModel) return;
  if (!r.ok) { grid.innerHTML = `<div class="result-empty" style="grid-column:1/-1">Representation service unreachable. (${esc(errMsg(r))})</div>`; return; }
  const d = r.data;
  // Only show the progress UI when a build is actually in flight for THIS focus.
  if (d.active_job) pollRepresentationJob(d.active_job.job_id);
  if (!d.built) {
    grid.innerHTML = `<div class="result-empty" style="grid-column:1/-1">No <b>${esc(model)}</b> model for <b>${esc(state.dataset)}</b> yet. Set params and click <b>Build representation</b>. (Ingest the dataset first.)</div>`;
    return;
  }
  const s = d.stats;
  applyRepControls(s);
  const built = (s.built_at || "").replace("T", " ").slice(0, 19);
  const fresh = s.cached ? "cached" : "fresh";
  let cards;
  if (model === "bert") {
    cards = [
      ["Documents", fmt(s.num_docs), "doc vectors"],
      ["Dimensions", fmt(s.dim), "BERT size"],
      ["Model", (s.params?.model_name || "").split("/").pop() || "—", "transformer"],
      ["Built at", built, fresh],
    ];
  } else if (model === "embedding") {
    cards = [
      ["Documents", fmt(s.num_docs), "doc vectors"],
      ["Dimensions", fmt(s.dim), "Word2Vec size"],
      ["Vocabulary", fmt(s.vocab_size), "words learned"],
      ["Built at", built, fresh],
    ];
  } else if (model === "bm25") {
    cards = [
      ["Documents", fmt(s.num_docs), "corpus size"],
      ["Vocabulary", fmt(s.vocab_size), "terms"],
      ["avgdl", (s.avgdl ?? 0).toFixed(2), "avg doc length"],
      ["min_df", fmt(s.params?.min_df ?? 1), "term pruning"],
      ["Built at", built, fresh],
    ];
  } else {
    cards = [
      ["Documents", fmt(s.num_docs), "matrix rows"],
      ["Vocabulary", fmt(s.vocab_size), "terms (columns)"],
      ["Non-zeros", fmt(s.nnz), "weights stored"],
      ["Density", ((s.density ?? 0) * 100).toFixed(4) + "%", "sparsity"],
      ["min_df / max_df", `${s.params?.min_df} / ${s.params?.max_df}`, s.params?.sublinear_tf ? "sublinear tf" : "raw tf"],
      ["Built at", built, fresh],
    ];
  }
  grid.innerHTML = cards.map((c) => `
    <div class="stat"><div class="s-label">${c[0]}</div><div class="s-val">${c[1]}</div><div class="s-sub">${esc(c[2])}</div></div>`).join("");
}

function repSyncControls() {
  const m = state.repModel;
  // Vocab params: TF-IDF (+min_df for BM25) only. Preprocessing: all but BERT (it reads raw text).
  $("#repParamsRow").hidden = (m === "embedding" || m === "bert");
  $("#repPpRow").hidden = (m === "bert");
  $$(".rep-tfidf-only").forEach((el) => { el.hidden = m !== "tfidf"; });
  const hints = {
    bm25: "k1 / b are tuned per query in the Search tab — not at build time. Build only stores corpus statistics.",
    embedding: "Word2Vec is trained offline on the corpus (skip-gram, 100-dim). A document's vector is the mean of its word vectors; build is a bit slower than the lexical models.",
    bert: "BERT (sentence-transformers) reads raw text — no preprocessing. The transformer downloads once (~90MB); this is the slowest build (CPU encoding).",
  };
  const hint = $("#repBuildHint");
  if (hints[m]) { hint.hidden = false; hint.textContent = hints[m]; } else hint.hidden = true;
}

async function buildRepresentation() {
  const model = state.repModel;
  const payload = { dataset: state.dataset, model, force: $("#repForce").checked, options: repPreprocessOptions() };
  if (model === "tfidf") payload.params = repParams();
  else if (model === "bm25") payload.bm25 = { min_df: Number($("#repMinDf").value) || 1 };
  // embedding: server defaults (transformer model name etc.)
  const r = await api("representation", "build", { method: "POST", json: payload });
  if (!r.ok) { toast(`build failed: ${errMsg(r)}`, "err", 6000); return; }
  const job = r.data;
  if (job.state === "skipped") { toast(`${model}: already built — enable “Force (rebuild)” to redo`, "info", 5000); loadRepresentation(); return; }
  toast(`${state.dataset}: building ${model} (full corpus)`, "info");
  pollRepresentationJob(job.job_id);
}

// Stop any poller and return the build controls to their idle state.
function resetRepProgress() {
  if (state.repPolling) { clearInterval(state.repPolling); state.repPolling = null; }
  const prog = $("#repProg"), lbl = $("#repProgLbl"), btn = $("#repBuild");
  if (prog) prog.hidden = true;
  if (lbl) lbl.hidden = true;
  if (btn) btn.disabled = false;
}

function pollRepresentationJob(jobId) {
  resetRepProgress();  // never stack pollers
  const prog = $("#repProg"), bar = $("#repBar"), lbl = $("#repProgLbl");
  prog.hidden = false; lbl.hidden = false; $("#repBuild").disabled = true;
  let fails = 0;
  state.repPolling = setInterval(async () => {
    const r = await api("representation", `jobs/${jobId}`);
    if (!r.ok) {
      // 404 ⇒ the job is gone (service restarted / evicted); other errors ⇒ a few
      // retries before giving up. Either way, never spin forever.
      if (r.status === 404 || ++fails >= 5) {
        resetRepProgress();
        toast("Lost track of the build — reloading status", "err", 4000);
        loadRepresentation();
      }
      return;
    }
    fails = 0;
    const j = r.data;
    const det = j.percent != null && j.percent > 0;
    prog.classList.toggle("indet", !det);
    if (det) bar.style.width = `${j.percent}%`;
    const counts = j.total ? `(${fmt(j.processed)}/${fmt(j.total)})` : `(${fmt(j.processed)})`;
    lbl.querySelector(".msg").textContent = `${j.message || j.state} ${counts}`;
    lbl.querySelector(".pct").innerHTML = det ? `${j.percent}%` : `<span class="spinner"></span>`;
    if (["succeeded", "failed", "skipped"].includes(j.state)) {
      resetRepProgress();
      toast(`${state.dataset}: ${state.repModel} ${j.state}${j.error ? " — " + j.error : ""}`, j.state === "failed" ? "err" : "ok", 6000);
      loadRepresentation();
      refreshRepresentStep();
    }
  }, 1000);
}

async function deleteRepresentation() {
  const model = state.repModel;
  if (!confirm(`Delete the ${model} representation for:\n\n${state.dataset}\n\n(the fitted model + matrix artifact). Rebuildable any time. Continue?`)) return;
  const r = await api("representation", `representation?dataset=${encodeURIComponent(state.dataset)}&model=${encodeURIComponent(model)}`, { method: "DELETE" });
  if (r.ok) toast(`${model} ${r.data.deleted ? "deleted" : "— none on disk"}`, "ok"); else toast(errMsg(r), "err", 5000);
  loadRepresentation();
  refreshRepresentStep();
}

async function inspectEncode() {
  const text = $("#repQuery").value.trim();
  const out = $("#repEncodeOut");
  if (state.repModel === "embedding" || state.repModel === "bert") {
    out.className = "result-empty";
    out.innerHTML = `Term weights are a <b>lexical</b> view (TF-IDF / BM25). Dense models (Word2Vec / BERT) have no term weights — use the <b>Search</b> tab to query them.`;
    return;
  }
  if (!text) { out.className = "result-empty"; out.textContent = "Enter a query."; return; }
  out.className = "loading"; out.innerHTML = `<span class="spinner"></span> encoding…`;
  const r = await api("representation", "encode", { method: "POST", json: { dataset: state.dataset, model: state.repModel, texts: [text], top_terms: 30 } });
  if (!r.ok) { out.className = "result-empty"; out.textContent = `Error: ${errMsg(r)} (build the model first?)`; return; }
  const v = r.data.vectors[0];
  out.className = "";
  if (!v || !v.terms.length) { out.innerHTML = `<p class="hint">No terms survived normalization, or none are in the vocabulary.</p>`; return; }
  const max = v.terms[0].weight || 1;
  out.innerHTML = `
    <div class="enc-meta">${fmt(v.nnz)} non-zero terms · vector dim ${fmt(r.data.dim)}</div>
    <div class="tws">${v.terms.map((t, i) => `
      <div class="tw"><span class="tw-rank">${i + 1}</span>
        <span class="tw-term">${esc(t.term)}</span>
        <span class="tw-track"><i style="width:${Math.max(2, (t.weight / max) * 100)}%"></i></span>
        <span class="tw-w">${t.weight.toFixed(4)}</span></div>`).join("")}</div>`;
}

async function refreshRepresentStep() {
  if (!state.dataset) return;
  const r = await api("representation", "status", { query: { dataset: state.dataset, model: "tfidf" } });
  setStep("represent", r.ok && r.data.built);
}

/* ============================================================
   SEARCH  (single model + hybrid serial/parallel + fusion)
   ============================================================ */
function seComponents() {
  return $$("#seParallel [data-comp]").filter((c) => c.checked).map((c) => c.dataset.comp);
}
function seUsesBm25() {
  if (state.seModel === "bm25") return true;
  if (state.seModel !== "hybrid") return false;
  return state.seMode === "parallel"
    ? seComponents().includes("bm25")
    : ($("#seFirst").value === "bm25" || $("#seRerank").value === "bm25");
}
function seSyncControls() {
  $("#seHybrid").hidden = state.seModel !== "hybrid";
  $("#seBm25Row").hidden = !seUsesBm25();
  if (state.seModel === "hybrid") {
    $("#seParallel").hidden = state.seMode !== "parallel";
    $("#seSerial").hidden = state.seMode !== "serial";
  }
}

async function runSearch() {
  const query = $("#seQuery").value.trim();
  const out = $("#seResults");
  if (!query) { toast("Enter a query", "err"); return; }
  const payload = {
    dataset: state.dataset, model: state.seModel, query, with_text: true,
    top_k: Number($("#seTopk").value) || 10,
    k1: Number($("#seK1").value), b: Number($("#seB").value),
  };
  if (state.seModel === "hybrid") {
    if (state.seMode === "parallel") {
      const comps = seComponents();
      if (comps.length < 2) { toast("Pick at least 2 components for parallel hybrid", "err", 4000); return; }
      payload.hybrid = { mode: "parallel", components: comps, fusion: state.seFusion, rrf_k: 60 };
    } else {
      const first = $("#seFirst").value, rerank = $("#seRerank").value;
      if (first === rerank) { toast("Serial needs two different models", "err", 4000); return; }
      payload.hybrid = { mode: "serial", first, rerank, candidates: Number($("#seCand").value) || 100 };
    }
  }
  out.className = "loading"; out.innerHTML = `<span class="spinner"></span> searching…`;
  $("#seMeta").hidden = true;
  const r = await api("representation", "search", { method: "POST", json: payload });
  if (!r.ok) { out.className = "result-empty"; out.textContent = `Error: ${errMsg(r)} (build the needed model(s) first?)`; return; }
  renderSearchResults(r.data);
}

function renderSearchResults(d) {
  const meta = $("#seMeta");
  meta.hidden = false;
  meta.textContent = `${fmt(d.total)} hits · ${d.model}${d.mode ? " · " + d.mode : ""} · ${d.took_ms} ms`;
  const out = $("#seResults");
  if (!d.hits.length) { out.className = "result-empty"; out.textContent = "No results — try another query or model."; return; }
  out.className = "";
  out.innerHTML = `<div class="hits">${d.hits.map((h) => `
    <div class="hit">
      <div class="hit-rank">${h.rank}</div>
      <div class="hit-body">
        <div class="hit-top">
          <span class="hit-id mono">${esc(h.doc_id)}</span>
          <span class="hit-score">${(h.score ?? 0).toFixed(4)}</span>
        </div>
        <div class="hit-text">${esc(trunc(h.text || "(original text unavailable)", 320))}</div>
      </div>
    </div>`).join("")}</div>`;
}

function wireSearch() {
  $("#seRun").addEventListener("click", runSearch);
  $("#seQuery").addEventListener("keydown", (e) => e.key === "Enter" && runSearch());
  // segmented controls
  $("#seModel").addEventListener("click", (e) => {
    const b = e.target.closest(".seg-btn"); if (!b) return;
    $$("#seModel .seg-btn").forEach((x) => x.classList.remove("active")); b.classList.add("active");
    state.seModel = b.dataset.m; seSyncControls();
  });
  $("#seMode").addEventListener("click", (e) => {
    const b = e.target.closest(".seg-btn"); if (!b) return;
    $$("#seMode .seg-btn").forEach((x) => x.classList.remove("active")); b.classList.add("active");
    state.seMode = b.dataset.mode; seSyncControls();
  });
  $("#seFusion").addEventListener("click", (e) => {
    const b = e.target.closest(".seg-btn"); if (!b) return;
    $$("#seFusion .seg-btn").forEach((x) => x.classList.remove("active")); b.classList.add("active");
    state.seFusion = b.dataset.f;
  });
  // bm25 sliders
  $("#seK1").addEventListener("input", (e) => { $("#seK1v").textContent = Number(e.target.value).toFixed(1); });
  $("#seB").addEventListener("input", (e) => { $("#seBv").textContent = Number(e.target.value).toFixed(2); });
  // re-evaluate whether BM25 params are relevant
  $("#seParallel").addEventListener("change", seSyncControls);
  $("#seSerial").addEventListener("change", seSyncControls);
}

/* ============================================================
   SERVICES
   ============================================================ */
function renderServicesSkeleton() {
  $("#svcGrid").innerHTML = state.meta.services.map((s) => `<div class="svc" id="svc-${cssId(s.key)}"></div>`).join("");
}

function renderServices() {
  const host = window.location.hostname;
  state.meta.services.forEach((s) => {
    const card = $(`#svc-${cssId(s.key)}`);
    if (!card) return;
    const h = state.health[s.key];
    const isUp = h === "up";
    const isInfra = s.group === "infra";
    const stateCls = isInfra ? "state-planned" : isUp ? "state-up" : (s.group === "planned" ? "state-planned" : "state-down");
    const stateTxt = isInfra ? "infra" : isUp ? "online" : (s.group === "planned" ? "planned" : "offline");
    card.className = `svc ${isUp ? "up" : isInfra ? "" : "down"}`;
    const docsLink = `http://${host}:${s.port}/docs`;
    const homeLink = `http://${host}:${s.port}/`;
    card.innerHTML = `
      <div class="svc-top">
        <span class="svc-dot"></span>
        <span class="svc-name">${esc(s.label)}</span>
        <span class="svc-state ${stateCls}">${stateTxt}</span>
      </div>
      <div class="svc-role">${esc(s.role)}</div>
      <div class="svc-meta"><span>${esc(s.key)}</span><span>·</span><span>:${s.port}</span></div>
      <div class="svc-links">
        ${isInfra
          ? `<a href="${homeLink}" target="_blank" class="btn btn-primary btn-sm">Open admin UI ↗</a>`
          : `<a href="${docsLink}" target="_blank" class="btn btn-ghost btn-sm">Swagger ↗</a>
             <a href="${homeLink}" target="_blank" class="btn btn-ghost btn-sm">Info ↗</a>`}
      </div>`;
  });
}

/* ============================================================
   WIRING
   ============================================================ */
function wireNav() {
  $$("#nav .nav-item").forEach((btn) => btn.addEventListener("click", () => {
    $$("#nav .nav-item").forEach((b) => b.classList.remove("active"));
    $$(".view").forEach((v) => v.classList.remove("active"));
    btn.classList.add("active");
    const view = btn.dataset.view;
    $(`.view[data-view='${view}']`).classList.add("active");
    onViewEnter(view);
  }));
}

function onViewEnter(view) {
  if (view === "datasets") renderDatasets();
  if (view === "index") loadIndexStats();
  if (view === "representation") loadRepresentation();
  if (view === "search") seSyncControls();
  if (view === "docstore") { browseDocs(true); browseQueries(true); }
  if (view === "services") renderServices();
  if (view === "overview") renderOverview();
}

function wireActions() {
  $("#refreshBtn").addEventListener("click", async () => { toast("Refreshing…", "info", 1200); await refreshAll(); });
  $("#datasetsRefresh").addEventListener("click", loadCatalog);

  // delegated dataset actions
  $("#datasetCards").addEventListener("click", (e) => {
    const b = e.target.closest("[data-act]");
    if (b) datasetAction(b.dataset.act, b.dataset.ds);
  });

  // preprocess
  $("#ppRun").addEventListener("click", runPreprocess);

  // docstore
  $("#dsFetch").addEventListener("click", fetchDocById);
  $("#dsDocId").addEventListener("keydown", (e) => e.key === "Enter" && fetchDocById());
  $("#dsPrev").addEventListener("click", () => { state.browse.offset = Math.max(0, state.browse.offset - state.browse.limit); browseDocs(); });
  $("#dsNext").addEventListener("click", () => { state.browse.offset += state.browse.limit; browseDocs(); });
  // test queries + qrels
  $("#qPrev").addEventListener("click", () => { state.qbrowse.offset = Math.max(0, state.qbrowse.offset - state.qbrowse.limit); browseQueries(); });
  $("#qNext").addEventListener("click", () => { state.qbrowse.offset += state.qbrowse.limit; browseQueries(); });
  $("#dsQueries").addEventListener("click", (e) => { const b = e.target.closest("[data-qid]"); if (b) loadQrelsForQuery(b.dataset.qid); });

  // index
  $("#ixReload").addEventListener("click", loadIndexStats);
  $("#ixNormalize").addEventListener("click", normalizeTerm);
  $("#ixLookup").addEventListener("click", lookupTerm);
  $("#ixTerm").addEventListener("keydown", (e) => e.key === "Enter" && lookupTerm());
  $("#ixDocLookup").addEventListener("click", lookupDoc);
  $("#ixDocId").addEventListener("keydown", (e) => e.key === "Enter" && lookupDoc());

  // search
  wireSearch();

  // representations
  $("#repReload").addEventListener("click", loadRepresentation);
  $("#repBuild").addEventListener("click", buildRepresentation);
  $("#repDelete").addEventListener("click", deleteRepresentation);
  $("#repEncode").addEventListener("click", inspectEncode);
  $("#repModels").addEventListener("click", (e) => {
    const b = e.target.closest(".model-card");
    if (!b || b.disabled) return;
    $$("#repModels .model-card").forEach((x) => x.classList.remove("active"));
    b.classList.add("active");
    state.repModel = b.dataset.model;
    loadRepresentation();
  });

  // global dataset focus
  $("#globalDataset").addEventListener("change", (e) => {
    state.dataset = e.target.value;
    const active = $(".view.active")?.dataset.view;
    onViewEnter(active);
    toast(`Focus → ${state.dataset}`, "info", 1500);
  });
}

function fillDatasetSelectors() {
  const sel = $("#globalDataset");
  sel.innerHTML = state.meta.datasets.map((d, i) => `<option value="${esc(d)}">${esc(d)}${i === 0 ? "  (primary)" : "  (bonus)"}</option>`).join("");
  sel.value = state.dataset || "";
}

function startClock() {
  const tick = () => { $("#clock").textContent = new Date().toLocaleTimeString(); };
  tick(); setInterval(tick, 1000);
}

/* ---- inline icons ---- */
const svgGrid = `<svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="3" width="7" height="7"/><rect x="14" y="3" width="7" height="7"/><rect x="14" y="14" width="7" height="7"/><rect x="3" y="14" width="7" height="7"/></svg>`;
const svgDb = `<svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" stroke-width="2"><ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5"/></svg>`;
const svgDoc = `<svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" stroke-width="2"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><path d="M14 2v6h6"/></svg>`;
const svgList = `<svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" stroke-width="2"><path d="M8 6h13M8 12h13M8 18h13M3 6h.01M3 12h.01M3 18h.01"/></svg>`;

document.addEventListener("DOMContentLoaded", boot);
