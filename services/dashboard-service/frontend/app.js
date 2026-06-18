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
  seBoolOp: "and",     // boolean (inverted-index) operator
  rfLast: null,        // last query-refinement response (for "search with refined")
  evReportId: null,    // evaluation report currently in view
  evCompareId: null,   // evaluation report to overlay (before/after compare)
  evReports: [],       // cached saved-reports list
  evPolling: null,     // evaluation job poller
  evRunType: "parallel", // custom-run builder type: parallel | serial | bm25sweep
  evFusion: "rrf",     // parallel fusion method in the builder
  evCustomRuns: [],    // custom run specs added in the builder (hybrids, BM25 sweep)
  evReportCache: null, // the currently-viewed report object (for export)
  evPq: null,          // cached per-query sidecar { report_id, runs:{label:{qid:{metric:val}}}, queries }
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
    tier("query path", byTier("query")), conn,
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
  const isBool = state.seModel === "boolean";
  $("#seHybrid").hidden = state.seModel !== "hybrid";
  $("#seBm25Row").hidden = isBool || !seUsesBm25();
  $("#seBoolRow").hidden = !isBool;
  if (state.seModel === "hybrid") {
    $("#seParallel").hidden = state.seMode !== "parallel";
    $("#seSerial").hidden = state.seMode !== "serial";
  }
}

async function runSearch() {
  let query = $("#seQuery").value.trim();
  const out = $("#seResults");
  if (!query) { toast("Enter a query", "err"); return; }
  const top_k = Number($("#seTopk").value) || 10;

  // Optional pre-retrieval refinement (the §6 "with vs without" toggle): refine the
  // raw query, then search with the refined one. Applies to every search mode.
  $("#seRefineNote").hidden = true;
  if ($("#seRefine").checked) {
    const ref = await refineQuery(query);
    if (ref) { query = ref.refined || query; showRefineNote(ref); }
  }

  // Boolean (inverted-index-only) search: a different endpoint and payload.
  if (state.seModel === "boolean") {
    out.className = "loading"; out.innerHTML = `<span class="spinner"></span> matching…`;
    $("#seMeta").hidden = true;
    const r = await api("retrieval", "boolean",
      { method: "POST", json: { dataset: state.dataset, query, operator: state.seBoolOp, top_k, with_text: true } });
    if (!r.ok) { out.className = "result-empty"; out.textContent = `Error: ${errMsg(r)} (build the index first?)`; return; }
    renderBooleanResults(r.data);
    return;
  }

  const payload = {
    dataset: state.dataset, model: state.seModel, query, with_text: true,
    top_k,
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
  const r = await api("retrieval", "search", { method: "POST", json: payload });
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

function renderBooleanResults(d) {
  const meta = $("#seMeta");
  meta.hidden = false;
  const terms = d.terms.map((t) => `${esc(t.term)}<span class="muted">(${fmt(t.df)})</span>`).join(", ") || "—";
  meta.innerHTML = `${fmt(d.total)} matched · <b>${esc(d.operator.toUpperCase())}</b> · index-only (no scoring) · terms: ${terms} · ${d.took_ms} ms`;
  const out = $("#seResults");
  if (!d.hits.length) { out.className = "result-empty"; out.textContent = "No documents match — try OR, fewer terms, or build the index."; return; }
  out.className = "";
  out.innerHTML = `<div class="hits">${d.hits.map((h) => `
    <div class="hit">
      <div class="hit-rank">${h.rank}</div>
      <div class="hit-body">
        <div class="hit-top">
          <span class="hit-id mono">${esc(h.doc_id)}</span>
          <span class="hit-score" title="distinct query terms matched">${h.matched} match${h.matched === 1 ? "" : "es"}</span>
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
  $("#seBoolOp").addEventListener("click", (e) => {
    const b = e.target.closest(".seg-btn"); if (!b) return;
    $$("#seBoolOp .seg-btn").forEach((x) => x.classList.remove("active")); b.classList.add("active");
    state.seBoolOp = b.dataset.op;
  });
  // bm25 sliders
  $("#seK1").addEventListener("input", (e) => { $("#seK1v").textContent = Number(e.target.value).toFixed(1); });
  $("#seB").addEventListener("input", (e) => { $("#seBv").textContent = Number(e.target.value).toFixed(2); });
  // re-evaluate whether BM25 params are relevant
  $("#seParallel").addEventListener("change", seSyncControls);
  $("#seSerial").addEventListener("change", seSyncControls);
}

/* ============================================================
   QUERY REFINEMENT  (spell-correct · expand · suggest)
   Standalone playground + the Search-tab "refine first" toggle.
   Talks to the query-refinement-service via the console proxy.
   ============================================================ */

// Refine a raw query using the Search-tab toggles; returns the response or null.
async function refineQuery(text) {
  const options = {
    correct_spelling: $("#seRefCorrect").checked,
    expand_synonyms: $("#seRefExpand").checked,
    max_synonyms_per_term: 2,
    max_edit_distance: 2,
  };
  const r = await api("query-refinement", "refine", { method: "POST", json: { text, options } });
  if (!r.ok) { toast(`Refinement failed: ${errMsg(r)} — searched the raw query`, "err", 4000); return null; }
  return r.data;
}

// Inline note above the search results summarising what refinement changed.
function showRefineNote(ref) {
  const el = $("#seRefineNote");
  el.hidden = false;
  const parts = [];
  if (ref.corrections?.length) parts.push("corrected " + ref.corrections.map((c) => `${esc(c.original)}→${esc(c.corrected)}`).join(", "));
  if (ref.added_terms?.length) parts.push("expanded +" + ref.added_terms.map(esc).join(", "));
  const summary = parts.length ? parts.join(" · ") : "no change";
  el.innerHTML = `refined → <b>${esc(ref.refined)}</b> <span class="muted">(${summary})</span>`;
}

async function runRefine() {
  const text = $("#rfText").value.trim();
  const out = $("#rfOut");
  if (!text) { toast("Enter a query", "err"); return; }
  const options = {
    correct_spelling: $("#rfCorrect").checked,
    expand_synonyms: $("#rfExpand").checked,
    protect_proper_nouns: $("#rfProtect").checked,
    max_synonyms_per_term: Number($("#rfMaxSyn").value) || 0,
    max_edit_distance: Number($("#rfMaxEdit").value) || 2,
  };
  out.className = "loading"; out.innerHTML = `<span class="spinner"></span> refining…`;
  $("#rfApplied").hidden = true;
  const r = await api("query-refinement", "refine", { method: "POST", json: { text, options } });
  if (!r.ok) {
    out.className = "result-empty";
    out.textContent = `Error: ${errMsg(r)} (is the query-refinement service up?)`;
    $("#rfToSearch").disabled = true; state.rfLast = null;
    return;
  }
  state.rfLast = r.data;
  $("#rfToSearch").disabled = !r.data.refined;
  renderRefineResult(r.data, out);
}

function renderRefineResult(d, out) {
  const applied = $("#rfApplied");
  applied.hidden = false;
  applied.textContent = d.applied?.length ? d.applied.join(" + ") : "no change";

  const corr = d.corrections?.length
    ? `<div class="chips">${d.corrections.map((c) => `<span class="chip"><s class="muted">${esc(c.original)}</s> → ${esc(c.corrected)}</span>`).join("")}</div>`
    : `<p class="hint">No spelling corrections.</p>`;
  const expandedOff = !$("#rfExpand").checked;
  const added = d.added_terms?.length
    ? `<div class="chips">${d.added_terms.map((t) => `<span class="chip">+ ${esc(t)}</span>`).join("")}</div>`
    : `<p class="hint">No synonyms added${expandedOff ? " (expansion off)" : ""}.</p>`;
  const sugg = d.suggestions?.length
    ? `<div class="chips">${d.suggestions.map((s) => `<span class="chip chip-link" data-sugg="${esc(s)}">${esc(s)}</span>`).join("")}</div>`
    : `<p class="hint">No alternative suggestion.</p>`;

  out.className = "";
  out.innerHTML = `
    <div class="rf-row"><span class="rf-k">original</span><div class="norm-box">${esc(d.original) || "<span style='color:var(--faint)'>(empty)</span>"}</div></div>
    <div class="rf-row"><span class="rf-k">refined</span><div class="norm-box">${esc(d.refined) || "<span style='color:var(--faint)'>(empty)</span>"}</div></div>
    <div class="rf-sec"><h4 class="rf-h">Corrections (did you mean)</h4>${corr}</div>
    <div class="rf-sec"><h4 class="rf-h">Added terms (synonyms)</h4>${added}</div>
    <div class="rf-sec"><h4 class="rf-h">Suggestions <span class="muted">— click to reuse</span></h4>${sugg}</div>`;
}

// "Search with refined" — push the refined query into the Search tab and run it.
function refineToSearch() {
  if (!state.rfLast?.refined) return;
  $("#seQuery").value = state.rfLast.refined;
  $("#seRefine").checked = false;  // already refined — don't refine twice
  $$("#nav .nav-item").forEach((b) => b.classList.remove("active"));
  $$(".view").forEach((v) => v.classList.remove("active"));
  $("#nav .nav-item[data-view='search']").classList.add("active");
  $(".view[data-view='search']").classList.add("active");
  onViewEnter("search");
  runSearch();
}

function wireRefine() {
  $("#rfRun").addEventListener("click", runRefine);
  $("#rfText").addEventListener("keydown", (e) => { if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) runRefine(); });
  $("#rfToSearch").addEventListener("click", refineToSearch);
  // suggestion chips → reuse in the input
  $("#rfOut").addEventListener("click", (e) => {
    const c = e.target.closest("[data-sugg]"); if (!c) return;
    $("#rfText").value = c.dataset.sugg;
  });
}

/* ============================================================
   EVALUATION  (run · save · view · charts · before/after)
   Talks to the evaluation-service via the console proxy. Charts are
   pure CSS bars (metrics are 0–1) — no library, works offline.
   ============================================================ */
const EV_BASE_METRICS = ["map", "ndcg@10", "recall@100", "precision@10"];
const EV_DEPTH_METRICS = ["ndcg@1", "ndcg@3", "ndcg@5", "ndcg@10", "ndcg@20", "recall@10", "recall@50", "recall@100", "recall@1000"];

function evSelectedModels() {
  return $$("[data-evm]").filter((c) => c.checked).map((c) => c.dataset.evm);
}

// Base single-model runs (checkboxes) + any custom runs added in the builder.
function evBuildRuns() {
  const runs = evSelectedModels().map((m) => ({ label: m, model: m }));
  for (const r of state.evCustomRuns) runs.push(r);
  return runs;
}

// Metrics sent to the backend: the four headline metrics, plus the @k depth families
// when "Depth sweep" is on (the backend computes any requested metric@k).
function evMetrics() {
  const set = [...EV_BASE_METRICS];
  if ($("#evDepthToggle") && $("#evDepthToggle").checked) for (const m of EV_DEPTH_METRICS) if (!set.includes(m)) set.push(m);
  return set;
}
function renderMetricChips() {
  const box = $("#evMetricChips");
  if (box) box.innerHTML = evMetrics().map((m) => `<span class="chip">${esc(m)}</span>`).join("");
}

const _parseNums = (s) => (s || "").split(",").map((x) => Number(x.trim())).filter((x) => Number.isFinite(x));

// Add run(s) to the builder from the active type's controls.
function evAddCustomRun() {
  const t = state.evRunType;
  let specs = [];
  if (t === "parallel") {
    const comps = $$("[data-evc]").filter((c) => c.checked).map((c) => c.dataset.evc);
    if (comps.length < 2) { toast("Parallel hybrid needs ≥ 2 components", "err", 4000); return; }
    const fusion = state.evFusion || "rrf";
    specs = [{ label: `${fusion}(${comps.join("+")})`, model: "hybrid", hybrid: { mode: "parallel", components: comps, fusion, rrf_k: 60 } }];
  } else if (t === "serial") {
    const first = $("#evFirst").value, rerank = $("#evRerank").value;
    if (first === rerank) { toast("Serial hybrid needs two different models", "err", 4000); return; }
    specs = [{ label: `serial(${first}→${rerank})`, model: "hybrid", hybrid: { mode: "serial", first, rerank, candidates: Number($("#evCand").value) || 100 } }];
  } else {
    const k1s = _parseNums($("#evK1s").value), bs = _parseNums($("#evBs").value);
    if (!k1s.length || !bs.length) { toast("Enter k1 and b values (comma-separated)", "err", 4000); return; }
    for (const k1 of k1s) for (const b of bs) specs.push({ label: `bm25(k1=${k1},b=${b})`, model: "bm25", k1, b });
  }
  let added = 0;
  for (const s of specs) {
    if (state.evCustomRuns.some((r) => r.label === s.label)) continue;
    state.evCustomRuns.push(s); added++;
  }
  renderCustomRunChips();
  toast(added ? `Added ${added} run(s)` : "Already added", added ? "ok" : "info", 1600);
}

function renderCustomRunChips() {
  const box = $("#evCustomRuns");
  if (!box) return;
  if (!state.evCustomRuns.length) {
    box.innerHTML = `<span class="hint" style="margin:0">No custom runs — the checked base models will be evaluated. Add hybrids or a BM25 sweep above.</span>`;
    return;
  }
  box.innerHTML = state.evCustomRuns.map((r, i) => `<span class="ev-chip-run">${esc(r.label)}<button data-rm="${i}" title="remove">×</button></span>`).join("");
}

async function runEvaluation() {
  const runs = evBuildRuns();
  if (!runs.length) { toast("Pick at least one base model or add a run", "err"); return; }
  const label = ($("#evLabel").value || "baseline").trim() || "baseline";
  const maxQ = $("#evMaxQ").value.trim();
  const payload = {
    dataset: state.dataset, label, runs, metrics: evMetrics(),
    top_k: Number($("#evTopk").value) || 100,
    force: $("#evForce").checked,
  };
  if (maxQ) payload.max_queries = Number(maxQ);
  // With/without refinement: when on, every test query is refined before retrieval, so
  // this report is the "after" — compare it to a baseline (no refine) report.
  if ($("#evRefine").checked) {
    payload.refine = {
      correct_spelling: $("#evRefCorrect").checked,
      expand_synonyms: $("#evRefExpand").checked,
      max_synonyms_per_term: 2,
      max_edit_distance: 2,
    };
  }

  const r = await api("evaluation", "evaluate", { method: "POST", json: payload });
  if (!r.ok) {
    // 409 with a job already in flight for this label → attach to it instead of erroring,
    // so the user sees its live progress rather than a dead toast.
    if (r.status === 409) {
      const running = await evResumeRunningJob();
      toast(running ? "An evaluation with this label is already running — showing its progress"
                    : `evaluation: ${errMsg(r)}`, running ? "info" : "err", 5000);
    } else {
      toast(`evaluation failed: ${errMsg(r)}`, "err", 6000);
    }
    return;
  }
  const job = r.data;
  // A fast skip (report already exists) may finish before the POST returns — handle both.
  if (["skipped", "succeeded", "failed"].includes(job.state)) { handleEvalDone(job); return; }
  toast(`Evaluating ${runs.length} run(s) on ${state.dataset}${maxQ ? ` · ${maxQ} queries` : " · all queries"}`, "info", 4500);
  pollEvalJob(job.job_id);
}

function resetEvalProgress() {
  if (state.evPolling) { clearInterval(state.evPolling); state.evPolling = null; }
  const prog = $("#evProg"), lbl = $("#evProgLbl"), btn = $("#evRun");
  if (prog) prog.hidden = true;
  if (lbl) lbl.hidden = true;
  if (btn) btn.disabled = false;
}

async function handleEvalDone(job) {
  resetEvalProgress();
  toast(`Evaluation ${job.state}${job.error ? " — " + job.error : ""}`, job.state === "failed" ? "err" : "ok", 6000);
  await loadEvalReports();
  const rid = job.result && job.result.report_id;
  if (rid) viewEvalReport(rid);
}

function pollEvalJob(jobId) {
  resetEvalProgress();
  const prog = $("#evProg"), bar = $("#evBar"), lbl = $("#evProgLbl");
  prog.hidden = false; lbl.hidden = false; $("#evRun").disabled = true;
  let fails = 0;
  state.evPolling = setInterval(async () => {
    const r = await api("evaluation", `jobs/${jobId}`);
    if (!r.ok) {
      if (r.status === 404 || ++fails >= 5) { resetEvalProgress(); toast("Lost track of the evaluation — reloading", "err", 4000); loadEvalReports(); }
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
    if (["succeeded", "failed", "skipped"].includes(j.state)) handleEvalDone(j);
  }, 1200);
}

// Attach the progress bar to an evaluation already running for the focused dataset —
// one started earlier, from another tab/session, or rejected just now with a 409
// "already in progress". So progress always shows, even if this tab didn't start it.
async function evResumeRunningJob() {
  const r = await api("evaluation", "jobs", { query: { dataset: state.dataset } });
  if (!r.ok || !Array.isArray(r.data)) return null;
  const running = r.data.find((j) => j.state === "running");
  if (running) pollEvalJob(running.job_id);
  return running || null;
}

// The dataset a saved report belongs to (from the cached list metadata).
function evReportDataset(rid) {
  const it = state.evReports.find((r) => r.report_id === rid);
  return it ? it.dataset_id : null;
}

// Reports for the dataset currently in focus (the Evaluation view is dataset-scoped,
// like every other tab — switching the sidebar focus re-filters everything here).
function evReportsForFocus() {
  return state.evReports.filter((it) => it.dataset_id === state.dataset);
}

async function loadEvalReports() {
  const box = $("#evReports");
  const badge = $("#evDsBadge");
  if (badge) badge.textContent = state.dataset || "—";
  const r = await api("evaluation", "reports");
  if (!r.ok) { box.className = "result-empty"; box.textContent = `Evaluation service unreachable. (${esc(errMsg(r))})`; return; }
  state.evReports = r.data.items || [];
  // If the open report (or its compare) is from another dataset, close the view — the
  // focus changed, so the previous dataset's table/charts no longer apply.
  if (state.evReportId && evReportDataset(state.evReportId) !== state.dataset) {
    state.evReportId = null; state.evCompareId = null;
    $("#evReportPanel").hidden = true;
  }
  renderEvalReports();
  populateCompareSelect();
}

function renderEvalReports() {
  const box = $("#evReports");
  const items = evReportsForFocus();
  const cnt = $("#evReportCount");
  cnt.hidden = !items.length;
  cnt.textContent = `${items.length} report${items.length === 1 ? "" : "s"}`;
  if (!items.length) {
    box.className = "result-empty";
    box.textContent = `No reports for ${state.dataset || "this dataset"} yet — run an evaluation above.`;
    return;
  }
  box.className = "ev-reports";
  box.innerHTML = items.map((it) => `
    <div class="ev-rep ${it.report_id === state.evReportId ? "sel" : ""}" data-rid="${esc(it.report_id)}">
      <div class="ev-rep-top">
        <span class="ev-rep-label">${esc(it.label)}</span>
        <span class="ev-rep-tag tag-primary">${esc(it.dataset_id.split("/").pop())}</span>
      </div>
      <div class="ev-rep-meta">${it.models.length} model(s) · ${fmt(it.num_queries)} queries<br>${(it.created_at || "").replace("T", " ").slice(0, 19)}</div>
      <div class="ev-rep-actions">
        <button class="btn btn-primary btn-sm" data-evact="view" data-rid="${esc(it.report_id)}">View</button>
        <button class="btn btn-danger btn-sm spacer" data-evact="delete" data-rid="${esc(it.report_id)}">Delete</button>
      </div>
    </div>`).join("");
}

function populateCompareSelect() {
  const sel = $("#evCompare");
  if (!sel) return;
  // Only same-dataset reports can be compared (a cross-dataset before/after is meaningless).
  const opts = ['<option value="">— none —</option>'].concat(
    evReportsForFocus()
      .filter((it) => it.report_id !== state.evReportId)
      .map((it) => `<option value="${esc(it.report_id)}" ${it.report_id === state.evCompareId ? "selected" : ""}>${esc(it.label)}</option>`)
  );
  sel.innerHTML = opts.join("");
}

async function fetchReport(rid) {
  const r = await api("evaluation", `report/${encodeURIComponent(rid)}`);
  return r.ok ? r.data : null;
}

async function viewEvalReport(rid) {
  state.evReportId = rid;
  renderEvalReports();  // refresh selection highlight
  const panel = $("#evReportPanel");
  panel.hidden = false;
  $("#evReportTable").innerHTML = `<div class="loading"><span class="spinner"></span> loading report…</div>`;
  $("#evCharts").innerHTML = "";
  const report = await fetchReport(rid);
  if (!report) { $("#evReportTable").innerHTML = `<div class="result-empty">Could not load report ${esc(rid)}.</div>`; return; }
  const compare = (state.evCompareId && state.evCompareId !== rid) ? await fetchReport(state.evCompareId) : null;
  populateCompareSelect();
  renderEvalReport(report, compare);
}

// The best (max) run label per metric — for highlighting the winner.
function evBestPerMetric(runs, metrics) {
  const best = {};
  for (const m of metrics) {
    let bv = -Infinity, bl = null;
    for (const run of runs) {
      if (run.error) continue;
      const v = run.metrics ? run.metrics[m] : null;
      if (v != null && v > bv) { bv = v; bl = run.label; }
    }
    best[m] = bl;
  }
  return best;
}

// Metrics shown in the table / bar charts / radar — the four headline ones (depth-sweep
// @k families go to their own line charts so the table stays readable).
function evHeadlineMetrics(report) {
  const h = (report.metrics || []).filter((m) => EV_BASE_METRICS.includes(m));
  return h.length ? h : (report.metrics || []);
}

function renderEvalReport(report, compare) {
  state.evReportCache = report;
  $("#evReportLabel").textContent = report.label;
  $("#evReportMeta").innerHTML =
    `${esc(report.dataset_id)} · ${report.runs.length} run(s) · ${fmt(report.num_queries)} judged queries · top_k ${report.top_k} · engine <b>${esc(report.engine)}</b> · ${(report.created_at || "").replace("T", " ").slice(0, 19)}`
    + (compare ? ` &nbsp;·&nbsp; Δ vs <b>${esc(compare.label)}</b>` : "");
  renderEvalTable(report, compare);
  renderEvalCharts(report, compare);
  renderEvalRadar(report);
  renderEvalDepth(report);
  renderEvalSweep(report);
  renderEvalLatency(report);
  setupPerQuery(report);
}

function renderEvalTable(report, compare) {
  const metrics = evHeadlineMetrics(report);
  const best = evBestPerMetric(report.runs, metrics);
  const cmp = {};
  if (compare) compare.runs.forEach((run) => { cmp[run.label] = run; });
  const head = `<tr><th>model</th>${metrics.map((m) => `<th>${esc(m)}</th>`).join("")}</tr>`;
  const rows = report.runs.map((run) => {
    if (run.error) return `<tr><td class="mono">${esc(run.label)}</td><td colspan="${metrics.length}" style="color:var(--bad)">⚠ ${esc(run.error)}</td></tr>`;
    const cells = metrics.map((m) => {
      const v = run.metrics ? run.metrics[m] : null;
      let cell = v == null ? "—" : v.toFixed(4);
      if (compare && v != null) {
        const cv = cmp[run.label] && cmp[run.label].metrics ? cmp[run.label].metrics[m] : null;
        if (cv != null) {
          const d = v - cv;
          const cls = d > 1e-6 ? "ev-up" : d < -1e-6 ? "ev-down" : "ev-flat";
          cell += ` <span class="ev-delta ${cls}">${d >= 0 ? "+" : ""}${d.toFixed(4)}</span>`;
        }
      }
      return `<td class="mono ${best[m] === run.label ? "metric-best" : ""}">${cell}</td>`;
    }).join("");
    return `<tr><td class="mono">${esc(run.label)}</td>${cells}</tr>`;
  }).join("");
  $("#evReportTable").innerHTML = `<table class="tbl ev-table"><thead>${head}</thead><tbody>${rows}</tbody></table>`;
}

function renderEvalCharts(report, compare) {
  const metrics = evHeadlineMetrics(report);
  const runs = report.runs.filter((r) => !r.error);
  const best = evBestPerMetric(report.runs, metrics);
  const cmp = {};
  if (compare) compare.runs.forEach((run) => { cmp[run.label] = run; });
  const legend = compare
    ? `<div class="ev-legend"><span><span class="sw sw-base"></span>${esc(report.label)}</span><span><span class="sw sw-cmp"></span>${esc(compare.label)}</span></div>`
    : "";
  const bar = (v, cls) => {
    const w = Math.max(2, Math.min(100, (v || 0) * 100));
    return `<div class="ev-bar-track ${cls}"><i style="width:${w}%"></i><span class="ev-bar-val">${(v || 0).toFixed(3)}</span></div>`;
  };
  const charts = metrics.map((m) => {
    const rows = runs.map((run) => {
      const v = run.metrics ? run.metrics[m] : 0;
      let bars = bar(v, "");
      if (compare) {
        const cv = cmp[run.label] && cmp[run.label].metrics ? cmp[run.label].metrics[m] : 0;
        bars += bar(cv, "cmp");
      }
      const isBest = best[m] === run.label && !compare;
      return `<div class="ev-bar-row ${isBest ? "best" : ""}"><span class="ev-bar-label">${esc(run.label)}</span><div class="ev-bar-group">${bars}</div></div>`;
    }).join("");
    return `<div class="ev-chart"><div class="ev-chart-head"><span class="ev-chart-title">${esc(m)}</span><span class="ev-chart-sub">0 – 1</span></div>${legend}<div class="ev-bars">${rows}</div></div>`;
  }).join("");
  $("#evCharts").innerHTML = charts || `<div class="result-empty">No successful runs to chart — check the table for per-model errors.</div>`;
}

/* ---- advanced charts (pure SVG / CSS, no library) ---- */
const EV_COLORS = ["#6d8cff", "#2fd98a", "#ffb43d", "#ff6178", "#9a7bff", "#36b3ff", "#e879f9", "#a3e635"];
const evColor = (i) => EV_COLORS[i % EV_COLORS.length];
const _hideSec = (id, yes) => { const el = $(id); if (el) el.hidden = yes; };

// Radar: each run as a polygon across the headline metrics (all 0–1).
function renderEvalRadar(report) {
  const metrics = evHeadlineMetrics(report);
  const runs = report.runs.filter((r) => !r.error);
  if (metrics.length < 3 || !runs.length) { _hideSec("#evRadarSec", true); return; }
  _hideSec("#evRadarSec", false);
  const size = 300, cx = size / 2, cy = size / 2, R = size / 2 - 48, N = metrics.length;
  const ang = (i) => -Math.PI / 2 + i * 2 * Math.PI / N;
  const pt = (i, r) => [cx + r * Math.cos(ang(i)), cy + r * Math.sin(ang(i))];
  let svg = `<svg viewBox="0 0 ${size} ${size}" width="${size}" height="${size}">`;
  for (const t of [0.25, 0.5, 0.75, 1]) {
    const pts = metrics.map((_, i) => pt(i, R * t).map((v) => v.toFixed(1)).join(",")).join(" ");
    svg += `<polygon points="${pts}" fill="none" stroke="#28304f" stroke-width="1"/>`;
  }
  metrics.forEach((m, i) => {
    const [x, y] = pt(i, R);
    svg += `<line x1="${cx}" y1="${cy}" x2="${x.toFixed(1)}" y2="${y.toFixed(1)}" stroke="#28304f"/>`;
    const [lx, ly] = pt(i, R + 15);
    const anchor = Math.abs(lx - cx) < 8 ? "middle" : lx > cx ? "start" : "end";
    svg += `<text x="${lx.toFixed(1)}" y="${ly.toFixed(1)}" fill="#949bc4" font-size="9.5" text-anchor="${anchor}" font-family="JetBrains Mono,monospace">${esc(m)}</text>`;
  });
  runs.forEach((run, ri) => {
    const c = evColor(ri);
    const pts = metrics.map((m, i) => pt(i, R * Math.max(0, Math.min(1, run.metrics[m] || 0))).map((v) => v.toFixed(1)).join(",")).join(" ");
    svg += `<polygon points="${pts}" fill="${c}22" stroke="${c}" stroke-width="2"/>`;
  });
  svg += `</svg>`;
  const legend = runs.map((run, ri) => `<div class="ri"><span class="rsw" style="background:${evColor(ri)}"></span>${esc(run.label)}</div>`).join("");
  $("#evRadar").innerHTML = `<div class="ev-radar-wrap">${svg}<div class="ev-radar-legend">${legend}</div></div>`;
}

// Group metrics into metric@k families with ≥2 cutoffs (e.g. ndcg@{1,3,5,10,20}).
function _families(metrics) {
  const groups = {};
  for (const m of metrics) {
    const at = m.indexOf("@");
    if (at < 0) continue;
    const base = m.slice(0, at), k = Number(m.slice(at + 1));
    if (!Number.isFinite(k)) continue;
    (groups[base] = groups[base] || []).push(k);
  }
  return Object.entries(groups).filter(([, ks]) => ks.length >= 2)
    .map(([base, ks]) => [base, ks.sort((a, b) => a - b)]);
}

function _lineChart(title, cutoffs, runs, valueAt) {
  const w = 340, h = 220, pl = 38, pr = 12, ptp = 14, pb = 28, iw = w - pl - pr, ih = h - ptp - pb;
  const xs = cutoffs.map((_, i) => pl + (cutoffs.length === 1 ? iw / 2 : i * iw / (cutoffs.length - 1)));
  const yOf = (v) => ptp + ih * (1 - Math.max(0, Math.min(1, v)));
  let svg = `<svg viewBox="0 0 ${w} ${h}" width="100%" height="${h}">`;
  for (const gy of [0, 0.25, 0.5, 0.75, 1]) {
    const y = yOf(gy).toFixed(1);
    svg += `<line x1="${pl}" y1="${y}" x2="${w - pr}" y2="${y}" stroke="#28304f"/>`;
    svg += `<text x="${pl - 5}" y="${(yOf(gy) + 3).toFixed(1)}" fill="#69719a" font-size="9" text-anchor="end" font-family="JetBrains Mono,monospace">${gy}</text>`;
  }
  cutoffs.forEach((k, i) => { svg += `<text x="${xs[i].toFixed(1)}" y="${h - 9}" fill="#949bc4" font-size="9" text-anchor="middle" font-family="JetBrains Mono,monospace">@${k}</text>`; });
  runs.forEach((run, ri) => {
    const c = evColor(ri);
    const poly = cutoffs.map((k, i) => `${xs[i].toFixed(1)},${yOf(valueAt(run, k)).toFixed(1)}`).join(" ");
    svg += `<polyline points="${poly}" fill="none" stroke="${c}" stroke-width="2"/>`;
    cutoffs.forEach((k, i) => { svg += `<circle cx="${xs[i].toFixed(1)}" cy="${yOf(valueAt(run, k)).toFixed(1)}" r="2.6" fill="${c}"/>`; });
  });
  svg += `</svg>`;
  return `<div class="ev-chart"><div class="ev-chart-head"><span class="ev-chart-title">${esc(title)}</span><span class="ev-chart-sub">0 – 1</span></div>${svg}</div>`;
}

function renderEvalDepth(report) {
  const runs = report.runs.filter((r) => !r.error);
  const fams = _families(report.metrics);
  if (!fams.length || !runs.length) { _hideSec("#evDepthSec", true); return; }
  _hideSec("#evDepthSec", false);
  const legend = `<div class="ev-legend" style="grid-column:1/-1">${runs.map((r, i) => `<span><span class="sw" style="background:${evColor(i)}"></span>${esc(r.label)}</span>`).join("")}</div>`;
  const charts = fams.map(([base, ks]) => _lineChart(`${base}@k`, ks, runs, (run, k) => run.metrics[`${base}@${k}`] || 0));
  $("#evDepth").innerHTML = legend + charts.join("");
}

const _lerp = (a, b, t) => Math.round(a + (b - a) * t);
function evHeatColor(t) {  // worse → better : dark slate → bright green
  const lo = [34, 45, 80], hi = [47, 217, 138];
  return `rgb(${_lerp(lo[0], hi[0], t)},${_lerp(lo[1], hi[1], t)},${_lerp(lo[2], hi[2], t)})`;
}

// BM25 k1/b sweep → heatmap (only when ≥2 bm25 runs vary k1 or b).
function renderEvalSweep(report) {
  const bm = report.runs.filter((r) => !r.error && r.model === "bm25" && r.params && r.params.k1 != null && r.params.b != null);
  const k1s = [...new Set(bm.map((r) => r.params.k1))].sort((a, b) => a - b);
  const bs = [...new Set(bm.map((r) => r.params.b))].sort((a, b) => a - b);
  if (bm.length < 2 || (k1s.length < 2 && bs.length < 2)) { _hideSec("#evSweepSec", true); return; }
  _hideSec("#evSweepSec", false);
  const hm = evHeadlineMetrics(report);
  const metric = hm.includes("ndcg@10") ? "ndcg@10" : hm[0];
  const cell = {}; let best = null; const vals = [];
  bm.forEach((r) => {
    const v = r.metrics[metric] || 0;
    cell[`${r.params.k1}|${r.params.b}`] = v; vals.push(v);
    if (!best || v > best.v) best = { v, k1: r.params.k1, b: r.params.b };
  });
  const lo = Math.min(...vals), hi = Math.max(...vals), span = (hi - lo) || 1;
  let html = `<div class="ev-heat" style="grid-template-columns: auto repeat(${k1s.length}, 1fr)">`;
  html += `<div class="ev-heat-axis ev-heat-corner">b \\ k1</div>`;
  k1s.forEach((k1) => { html += `<div class="ev-heat-axis">${k1}</div>`; });
  bs.forEach((b) => {
    html += `<div class="ev-heat-axis">${b}</div>`;
    k1s.forEach((k1) => {
      const v = cell[`${k1}|${b}`];
      if (v == null) { html += `<div class="ev-heat-cell" style="background:var(--surface-2);color:var(--faint)">—</div>`; return; }
      const isBest = best && k1 === best.k1 && b === best.b;
      html += `<div class="ev-heat-cell ${isBest ? "ev-heat-best" : ""}" style="background:${evHeatColor((v - lo) / span)};color:#fff;text-shadow:0 1px 2px rgba(0,0,0,.7)">${v.toFixed(3)}</div>`;
    });
  });
  html += `</div><div class="ev-heat-note">Cell = <b>${esc(metric)}</b> (slate→green = worse→better). Best: <b>k1=${best.k1}, b=${best.b}</b> → ${best.v.toFixed(4)}.</div>`;
  $("#evSweep").innerHTML = html;
}

// Quality vs speed scatter (needs avg_query_ms). Names live in a side legend (not on the
// plot) so points never overlap their labels; hover a dot for exact values.
function renderEvalLatency(report) {
  const runs = report.runs.filter((r) => !r.error && r.avg_query_ms != null);
  if (!runs.length) { _hideSec("#evLatencySec", true); return; }
  _hideSec("#evLatencySec", false);
  const hm = evHeadlineMetrics(report);
  const metric = hm.includes("ndcg@10") ? "ndcg@10" : hm[0];
  const W = 520, H = 320, pl = 54, pr = 18, pt = 18, pb = 50, iw = W - pl - pr, ih = H - pt - pb;
  const xMax = (Math.max(...runs.map((r) => r.avg_query_ms)) * 1.12) || 1;
  const xOf = (ms) => pl + iw * (ms / xMax);
  const yOf = (v) => pt + ih * (1 - Math.max(0, Math.min(1, v)));
  const best = runs.reduce((a, r) => ((r.metrics[metric] || 0) > (a.metrics[metric] || 0) ? r : a), runs[0]);
  let g = "";
  // horizontal grid + y labels (the quality axis, 0–1)
  for (const t of [0, 0.2, 0.4, 0.6, 0.8, 1]) {
    const y = yOf(t).toFixed(1);
    g += `<line x1="${pl}" y1="${y}" x2="${W - pr}" y2="${y}" stroke="#28304f"/>`;
    g += `<text x="${pl - 8}" y="${(yOf(t) + 3).toFixed(1)}" fill="#69719a" font-size="10" text-anchor="end" font-family="JetBrains Mono,monospace">${t.toFixed(1)}</text>`;
  }
  // vertical grid + x labels (the latency axis, ms)
  for (let i = 0; i <= 5; i++) {
    const ms = xMax * i / 5, x = xOf(ms).toFixed(1);
    g += `<line x1="${x}" y1="${pt}" x2="${x}" y2="${pt + ih}" stroke="#202746"/>`;
    g += `<text x="${x}" y="${H - pb + 17}" fill="#69719a" font-size="10" text-anchor="middle" font-family="JetBrains Mono,monospace">${ms.toFixed(0)}</text>`;
  }
  // axis titles with direction hints
  g += `<text x="${(pl + iw / 2).toFixed(1)}" y="${H - 8}" fill="#c7cdec" font-size="11" text-anchor="middle">latency · ms / query  (← faster)</text>`;
  const my = (pt + ih / 2).toFixed(1);
  g += `<text x="15" y="${my}" fill="#c7cdec" font-size="11" text-anchor="middle" transform="rotate(-90 15 ${my})">${esc(metric)}  (↑ better)</text>`;
  g += `<text x="${pl + 6}" y="${pt + 13}" fill="#2fd98a" font-size="10" font-family="JetBrains Mono,monospace">▲ ideal corner</text>`;
  // points (colored dots only — labels are in the legend; <title> = hover tooltip)
  runs.forEach((run, i) => {
    const c = evColor(i), x = xOf(run.avg_query_ms).toFixed(1), y = yOf(run.metrics[metric] || 0).toFixed(1);
    const isBest = run === best;
    g += `<circle cx="${x}" cy="${y}" r="${isBest ? 7 : 5.5}" fill="${c}" stroke="${isBest ? "#fff" : "#0a0c16"}" stroke-width="${isBest ? 2 : 1.5}"><title>${esc(run.label)} — ${run.avg_query_ms.toFixed(0)} ms · ${esc(metric)} ${(run.metrics[metric] || 0).toFixed(4)}</title></circle>`;
  });
  const svg = `<svg viewBox="0 0 ${W} ${H}" width="100%" height="${H}" preserveAspectRatio="xMidYMid meet">${g}</svg>`;
  const legend = runs.map((run, i) => `<div class="ri"><span class="rsw" style="background:${evColor(i)}"></span><span>${esc(run.label)}</span><span class="ev-leg-v">${run.avg_query_ms.toFixed(0)}ms · ${(run.metrics[metric] || 0).toFixed(3)}</span></div>`).join("");
  $("#evLatency").innerHTML = `<div class="ev-scatter-wrap"><div class="ev-scatter-plot">${svg}</div><div class="ev-radar-legend ev-scatter-legend">${legend}</div></div>
    <div class="ev-heat-note">Each dot is a run. <b>Top-left = best</b> (high ${esc(metric)}, low latency). Hover a dot for exact values · best ${esc(metric)}: <b>${esc(best.label)}</b>.</div>`;
}

/* ---- per-query drill-down (best/worst + head-to-head) ---- */
function setupPerQuery(report) {
  state.evPq = null;
  const runs = report.runs.filter((r) => !r.error).map((r) => r.label);
  const box = $("#evPq");
  if (!runs.length) { box.innerHTML = `<div class="result-empty">No successful runs to analyze.</div>`; return; }
  const metrics = evHeadlineMetrics(report);
  const opt = (arr, sel) => arr.map((x) => `<option value="${esc(x)}" ${x === sel ? "selected" : ""}>${esc(x)}</option>`).join("");
  box.innerHTML = `
    <div class="ev-pq-controls">
      <label class="opt-field">run <select id="evPqRun" class="select sm">${opt(runs, runs[0])}</select></label>
      <label class="opt-field">vs <select id="evPqRunB" class="select sm">${opt(runs, runs[1] || runs[0])}</select></label>
      <label class="opt-field">metric <select id="evPqMetric" class="select sm">${opt(metrics, metrics.includes("ndcg@10") ? "ndcg@10" : metrics[0])}</select></label>
      <button class="btn btn-primary btn-sm" id="evPqLoad" data-rid="${esc(report.report_id)}">Load analysis</button>
    </div>
    <div id="evPqOut" class="result-empty">Load the per-query breakdown to see best/worst queries and head-to-head wins.</div>`;
  $("#evPqLoad").addEventListener("click", (e) => loadPerQuery(e.currentTarget.dataset.rid));
  ["evPqRun", "evPqRunB", "evPqMetric"].forEach((id) => $("#" + id).addEventListener("change", () => { if (state.evPq) renderPerQuery(); }));
}

async function loadPerQuery(reportId) {
  const out = $("#evPqOut");
  out.className = "loading"; out.innerHTML = `<span class="spinner"></span> loading per-query data…`;
  const r = await api("evaluation", `report/${encodeURIComponent(reportId)}/per-query`);
  if (!r.ok) { out.className = "result-empty"; out.textContent = `No per-query data: ${esc(errMsg(r))} (re-run with per_query enabled).`; return; }
  state.evPq = r.data;
  renderPerQuery();
}

function renderPerQuery() {
  const pq = state.evPq; if (!pq) return;
  const a = $("#evPqRun").value, b = $("#evPqRunB").value, m = $("#evPqMetric").value;
  const ra = pq.runs[a] || {}, rb = pq.runs[b] || {};
  const qtext = (qid) => trunc(pq.queries[qid] || "", 90);
  const rows = Object.keys(ra).map((qid) => ({ qid, v: ra[qid][m] ?? 0 })).sort((x, y) => x.v - y.v);
  const worst = rows.slice(0, 8), best = rows.slice(-8).reverse();
  const qtbl = (title, list) => `
    <div class="ev-chart">
      <div class="ev-chart-head"><span class="ev-chart-title">${title}</span><span class="ev-chart-sub">${esc(m)}</span></div>
      <table class="tbl"><thead><tr><th style="width:70px">query</th><th>text</th><th style="width:58px">${esc(m)}</th></tr></thead>
      <tbody>${list.map((r) => `<tr><td class="mono">${esc(r.qid)}</td><td class="ev-q">${esc(qtext(r.qid))}</td><td class="mono">${r.v.toFixed(3)}</td></tr>`).join("")}</tbody></table>
    </div>`;
  let win = 0, tie = 0, loss = 0; const diffs = [];
  if (a !== b) {
    for (const qid of Object.keys(ra)) {
      if (!(qid in rb)) continue;
      const d = (ra[qid][m] ?? 0) - (rb[qid][m] ?? 0);
      if (d > 1e-6) win++; else if (d < -1e-6) loss++; else tie++;
      diffs.push({ qid, d, av: ra[qid][m] ?? 0, bv: rb[qid][m] ?? 0 });
    }
    diffs.sort((x, y) => Math.abs(y.d) - Math.abs(x.d));
  }
  const top = diffs.slice(0, 10);
  const wl = a === b ? `<p class="hint">Pick a different run in “vs” for a head-to-head comparison.</p>` : `
    <h4 class="ev-sec">${esc(a)} vs ${esc(b)} · head-to-head on ${esc(m)} (${win + tie + loss} queries)</h4>
    <div class="ev-winloss">
      <div class="ev-wl win"><div class="v">${win}</div><div class="k">${esc(a)} wins</div></div>
      <div class="ev-wl tie"><div class="v">${tie}</div><div class="k">tie</div></div>
      <div class="ev-wl loss"><div class="v">${loss}</div><div class="k">${esc(b)} wins</div></div>
    </div>
    <table class="tbl"><thead><tr><th style="width:70px">query</th><th>text</th><th style="width:56px">${esc(a)}</th><th style="width:56px">${esc(b)}</th><th style="width:64px">Δ</th></tr></thead>
    <tbody>${top.map((r) => `<tr><td class="mono">${esc(r.qid)}</td><td class="ev-q">${esc(qtext(r.qid))}</td><td class="mono">${r.av.toFixed(3)}</td><td class="mono">${r.bv.toFixed(3)}</td><td class="mono"><span class="ev-delta ${r.d > 0 ? "ev-up" : r.d < 0 ? "ev-down" : "ev-flat"}">${r.d >= 0 ? "+" : ""}${r.d.toFixed(3)}</span></td></tr>`).join("")}</tbody></table>`;
  $("#evPqOut").className = "";
  $("#evPqOut").innerHTML = `${wl}
    <h4 class="ev-sec">Where ${esc(a)} struggles / shines · ${esc(m)}</h4>
    <div class="ev-charts">${qtbl("Worst queries", worst)}${qtbl("Best queries", best)}</div>`;
}

/* ---- export ---- */
function _download(name, text, type) {
  const url = URL.createObjectURL(new Blob([text], { type }));
  const a = document.createElement("a");
  a.href = url; a.download = name; a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
function exportEvalCsv() {
  const rep = state.evReportCache; if (!rep) { toast("Open a report first", "info"); return; }
  const lines = ["run_label,model,mode,metric,value"];
  for (const run of rep.runs) {
    if (run.error) continue;
    for (const m of rep.metrics) if (run.metrics[m] != null) lines.push(`${run.label},${run.model},${run.mode || ""},${m},${run.metrics[m]}`);
  }
  _download(`${rep.report_id}.csv`, lines.join("\n"), "text/csv");
}
function exportEvalJson() {
  const rep = state.evReportCache; if (!rep) { toast("Open a report first", "info"); return; }
  _download(`${rep.report_id}.json`, JSON.stringify(rep, null, 2), "application/json");
}

async function deleteEvalReport(rid) {
  if (!confirm(`Delete evaluation report:\n\n${rid}\n\n(its JSON + CSV). Continue?`)) return;
  const r = await api("evaluation", `report/${encodeURIComponent(rid)}`, { method: "DELETE" });
  if (r.ok) toast(`Report ${r.data.deleted ? "deleted" : "— not found"}`, "ok"); else toast(errMsg(r), "err", 5000);
  if (state.evReportId === rid) { state.evReportId = null; $("#evReportPanel").hidden = true; }
  if (state.evCompareId === rid) state.evCompareId = null;
  await loadEvalReports();
}

function evSyncRunType() {
  $$(".ev-bt").forEach((el) => { el.hidden = el.dataset.bt !== state.evRunType; });
}

// One-time-per-enter init of the builder controls (chips + metric chips + type rows).
function initEvalControls() {
  renderCustomRunChips();
  renderMetricChips();
  evSyncRunType();
}

function wireEvaluation() {
  $("#evReload").addEventListener("click", loadEvalReports);
  $("#evRun").addEventListener("click", runEvaluation);
  // Nudge the label toward a distinct one when refinement is on, so the "after" run
  // doesn't overwrite the "baseline" report (and the two stay comparable).
  $("#evRefine").addEventListener("change", (e) => {
    const lbl = $("#evLabel");
    if (e.target.checked && lbl.value.trim() === "baseline") lbl.value = "with-refinement";
    else if (!e.target.checked && lbl.value.trim() === "with-refinement") lbl.value = "baseline";
  });
  $("#evReports").addEventListener("click", (e) => {
    const b = e.target.closest("[data-evact]");
    if (!b) return;
    if (b.dataset.evact === "view") viewEvalReport(b.dataset.rid);
    else if (b.dataset.evact === "delete") deleteEvalReport(b.dataset.rid);
  });
  $("#evCompare").addEventListener("change", (e) => {
    state.evCompareId = e.target.value || null;
    if (state.evReportId) viewEvalReport(state.evReportId);
  });

  // runs builder
  $("#evRunType").addEventListener("click", (e) => {
    const b = e.target.closest(".seg-btn"); if (!b) return;
    $$("#evRunType .seg-btn").forEach((x) => x.classList.remove("active")); b.classList.add("active");
    state.evRunType = b.dataset.t; evSyncRunType();
  });
  $("#evFusion").addEventListener("click", (e) => {
    const b = e.target.closest(".seg-btn"); if (!b) return;
    $$("#evFusion .seg-btn").forEach((x) => x.classList.remove("active")); b.classList.add("active");
    state.evFusion = b.dataset.f;
  });
  $("#evAddRun").addEventListener("click", evAddCustomRun);
  $("#evCustomRuns").addEventListener("click", (e) => {
    const b = e.target.closest("[data-rm]"); if (!b) return;
    state.evCustomRuns.splice(Number(b.dataset.rm), 1); renderCustomRunChips();
  });
  $("#evDepthToggle").addEventListener("change", renderMetricChips);

  // export
  $("#evExportCsv").addEventListener("click", exportEvalCsv);
  $("#evExportJson").addEventListener("click", exportEvalJson);
}

/* ============================================================
   SERVICES
   ============================================================ */
function renderServicesSkeleton() {
  $("#svcGrid").innerHTML = state.meta.services.map((s) => `<div class="svc" id="svc-${cssId(s.key)}"></div>`).join("");
}

function renderServices() {
  const host = window.location.hostname;
  // Internal services aren't reachable on the host; deep-link their Swagger through
  // the gateway's grouped docs (tag === service key). The gateway itself IS published.
  const gwPort = (state.meta.services.find((x) => x.key === "gateway") || {}).port || 8000;
  state.meta.services.forEach((s) => {
    const card = $(`#svc-${cssId(s.key)}`);
    if (!card) return;
    const h = state.health[s.key];
    const isUp = h === "up";
    const isInfra = s.group === "infra";
    const stateCls = isInfra ? "state-planned" : isUp ? "state-up" : (s.group === "planned" ? "state-planned" : "state-down");
    const stateTxt = isInfra ? "infra" : isUp ? "online" : (s.group === "planned" ? "planned" : "offline");
    card.className = `svc ${isUp ? "up" : isInfra ? "" : "down"}`;
    // External (published) nodes link straight to their own port; internal services
    // are only reachable via the gateway, so their Swagger opens the gateway's docs
    // anchored at this service's tag.
    let links;
    if (isInfra) {
      links = `<a href="http://${host}:${s.port}/" target="_blank" class="btn btn-primary btn-sm">Open admin UI ↗</a>`;
    } else if (s.external) {
      links = `<a href="http://${host}:${s.port}/docs" target="_blank" class="btn btn-ghost btn-sm">Swagger ↗</a>
               <a href="http://${host}:${s.port}/" target="_blank" class="btn btn-ghost btn-sm">Info ↗</a>`;
    } else {
      links = `<a href="http://${host}:${gwPort}/docs#/${s.key}" target="_blank" class="btn btn-ghost btn-sm">Swagger (gateway) ↗</a>`;
    }
    card.innerHTML = `
      <div class="svc-top">
        <span class="svc-dot"></span>
        <span class="svc-name">${esc(s.label)}</span>
        <span class="svc-state ${stateCls}">${stateTxt}</span>
      </div>
      <div class="svc-role">${esc(s.role)}</div>
      <div class="svc-meta"><span>${esc(s.key)}</span><span>·</span><span>:${s.port}</span></div>
      <div class="svc-links">${links}</div>`;
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
  if (view === "evaluation") { initEvalControls(); loadEvalReports(); evResumeRunningJob(); }
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

  // query refinement (standalone playground)
  wireRefine();

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

  // evaluation
  wireEvaluation();

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
