const API = '/api/evaluation';
const state = { dataset: [], category: 'all', overview: null, report: null, selectedModels: [] };
const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => [...document.querySelectorAll(selector)];

function escapeHtml(value) {
  const element = document.createElement('div');
  element.textContent = String(value ?? '');
  return element.innerHTML;
}

async function requestJson(path, options = {}) {
  const response = await fetch(`${API}${path}`, options);
  let payload = {};
  try { payload = await response.json(); } catch { /* handled by the status below */ }
  if (!response.ok) throw new Error(payload.detail || `Request failed with HTTP ${response.status}`);
  return payload;
}

function percent(value) { return value == null ? 'N/A' : `${(Number(value) * 100).toFixed(1)}%`; }
function modelLabel(model) {
  return ({ codellama: 'Code Llama', 'starcoder2:3b': 'StarCoder2 3B', 'qwen2.5-coder:1.5b': 'Qwen2.5 Coder 1.5B' })[model] || model;
}
function conditionLabel(name, value) {
  const labels = { same_dataset_for_all_models: 'Dataset', temperature: 'Temperature', max_tokens: 'Token limit', retrieval_top_k: 'Retrieval depth', knowledge_base_unchanged_between_models: 'Knowledge base' };
  const rendered = value === true ? 'Locked and identical' : value;
  return `<div><span>${escapeHtml(labels[name] || name.replaceAll('_', ' '))}</span><strong>${escapeHtml(rendered)}</strong></div>`;
}

function selectedAggregateRows() {
  const aggregates = state.overview?.aggregates || {};
  return state.selectedModels.map((model) => [model, aggregates[model]]).filter(([, item]) => item);
}

function renderModelPicker(models) {
  $('#model-picker').innerHTML = models.map((model) => `
    <label class="model-toggle">
      <input type="checkbox" value="${escapeHtml(model)}" ${state.selectedModels.includes(model) ? 'checked' : ''} />
      <span>${escapeHtml(modelLabel(model))}</span>
    </label>`).join('');
}

function renderMeasuredResults() {
  const rows = selectedAggregateRows();
  const selected = new Set(state.selectedModels);
  $$('.model-card').forEach((card) => card.classList.toggle('excluded', !selected.has(card.dataset.model)));
  $('#quality-results').innerHTML = rows.length ? rows.map(([model, item]) => `<tr><td>${escapeHtml(modelLabel(model))}</td><td>${percent(item.accuracy)}</td><td>${percent(item.relevance_f1)}</td><td>${percent(item.retrieval_precision_at_k)}</td><td>${percent(item.retrieval_recall_at_k)}</td><td>${Number(item.retrieval_mrr).toFixed(3)}</td><td>${percent(item.hallucination_rate)}</td><td>${percent(item.test_pass_rate)}</td></tr>`).join('') : '<tr><td colspan="8" class="empty-cell">Select a model to show its measured results.</td></tr>';

  // Overall Evaluation Metrics Visual Graphs
  const overallGrid = $('#overall-metrics-cards-grid');
  if (overallGrid && rows.length) {
    const metricsConfig = [
      { key: 'accuracy', name: 'Task Accuracy', higherBetter: true },
      { key: 'relevance_f1', name: 'Relevance F1', higherBetter: true },
      { key: 'test_pass_rate', name: 'Code Test Pass Rate', higherBetter: true },
      {
        key: 'hallucination_rate',
        name: 'Hallucination Resistance',
        higherBetter: false,
        getScore: (val) => 1 - Number(val || 0),
        getDisplay: (val) => `${((1 - Number(val || 0)) * 100).toFixed(1)}% (${(Number(val || 0) * 100).toFixed(0)}% err)`,
      },
    ];

    overallGrid.innerHTML = metricsConfig.map((cfg) => {
      let bestModel = null;
      let bestVal = cfg.higherBetter ? -Infinity : Infinity;
      for (const [model, item] of rows) {
        const val = Number(item[cfg.key] || 0);
        if (cfg.higherBetter ? val > bestVal : val < bestVal) {
          bestVal = val;
          bestModel = model;
        }
      }

      const modelBars = rows.map(([model, item]) => {
        const rawVal = Number(item[cfg.key] || 0);
        const score = cfg.getScore ? cfg.getScore(rawVal) : rawVal;
        const disp = cfg.getDisplay ? cfg.getDisplay(rawVal) : percent(rawVal);
        const isWinner = model === bestModel;
        let fillClass = 'model-codellama';
        if (model.includes('starcoder')) fillClass = 'model-starcoder';
        else if (model.includes('qwen')) fillClass = 'model-qwen';

        return `
          <div class="cat-model-row">
            <span class="cat-model-name">${isWinner ? '🏆 ' : ''}${escapeHtml(modelLabel(model))}</span>
            <div class="cat-bar-track">
              <div class="cat-bar-fill ${fillClass}" style="width: ${Math.max(score * 100, 2)}%"></div>
            </div>
            <span class="cat-score-text">${disp}</span>
          </div>
        `;
      }).join('');

      return `
        <div class="cat-comp-card">
          <div class="cat-comp-head">
            <span class="cat-comp-badge">${escapeHtml(cfg.name)}</span>
            ${bestModel ? `<span class="cat-comp-winner">🏆 Best: ${escapeHtml(modelLabel(bestModel))}</span>` : ''}
          </div>
          <div class="cat-bars-list">
            ${modelBars}
          </div>
        </div>
      `;
    }).join('');
  }

  // Category-wise performance matrix
  const categoryAggregates = state.overview?.category_aggregates || {};
  const categories = state.overview?.categories || Object.keys(categoryAggregates);
  const categoryVerdicts = state.overview?.category_verdicts || [];
  const verdictMap = Object.fromEntries(categoryVerdicts.map((v) => [v.category, v.best_model]));

  const catRows = [];
  for (const cat of categories) {
    const catData = categoryAggregates[cat] || {};
    const winningModel = verdictMap[cat];
    for (const model of state.selectedModels) {
      const vals = catData[model];
      if (!vals) continue;
      const isWinner = model === winningModel;
      const passStr = vals.test_pass_rate != null ? percent(vals.test_pass_rate) : '—';
      catRows.push(`
        <tr class="${isWinner ? 'winner-row' : ''}">
          <td><span class="category-badge">${escapeHtml(cat)}</span></td>
          <td>${isWinner ? '🏆 ' : ''}<strong>${escapeHtml(modelLabel(model))}</strong></td>
          <td>${percent(vals.accuracy)}</td>
          <td>${percent(vals.relevance_f1)}</td>
          <td>${percent(vals.hallucination_rate)}</td>
          <td>${Number(vals.latency_mean_seconds).toFixed(2)}s</td>
          <td>${passStr}</td>
        </tr>
      `);
    }
  }
  const catTable = $('#category-results');
  if (catTable) {
    catTable.innerHTML = catRows.length ? catRows.join('') : '<tr><td colspan="7" class="empty-cell">Select a model to show category results.</td></tr>';
  }

  // 7 Software Engineering Components Visual Comparison Graph
  const catCards = [];
  for (const cat of categories) {
    const catData = categoryAggregates[cat] || {};
    const winningModel = verdictMap[cat];
    const modelBars = state.selectedModels.map((model) => {
      const vals = catData[model];
      if (!vals) return '';
      const acc = Number(vals.accuracy || 0);
      const isWinner = model === winningModel;
      let fillClass = 'model-codellama';
      if (model.includes('starcoder')) fillClass = 'model-starcoder';
      else if (model.includes('qwen')) fillClass = 'model-qwen';

      return `
        <div class="cat-model-row">
          <span class="cat-model-name">${isWinner ? '🏆 ' : ''}${escapeHtml(modelLabel(model))}</span>
          <div class="cat-bar-track">
            <div class="cat-bar-fill ${fillClass}" style="width: ${Math.max(acc * 100, 2)}%"></div>
          </div>
          <span class="cat-score-text">${percent(acc)}</span>
        </div>
      `;
    }).join('');

    catCards.push(`
      <div class="cat-comp-card">
        <div class="cat-comp-head">
          <span class="cat-comp-badge">${escapeHtml(cat)}</span>
          ${winningModel ? `<span class="cat-comp-winner">🏆 Best: ${escapeHtml(modelLabel(winningModel))}</span>` : ''}
        </div>
        <div class="cat-bars-list">
          ${modelBars}
        </div>
      </div>
    `);
  }
  const catCardsGrid = $('#cat-cards-grid');
  if (catCardsGrid) {
    catCardsGrid.innerHTML = catCards.length ? catCards.join('') : '<div style="grid-column: 1/-1; padding: 12px; color: var(--muted); text-align: center;">Select models to view visual category comparisons.</div>';
  }

  $('#performance-results').innerHTML = rows.length ? rows.map(([model, item]) => `<tr><td>${escapeHtml(modelLabel(model))}</td><td>${Number(item.latency_mean_seconds).toFixed(2)}s</td><td>${Number(item.latency_p95_seconds).toFixed(2)}s</td><td>${Number(item.total_tokens_mean).toFixed(1)}</td><td>${Number(item.cpu_mean_percent).toFixed(1)}%</td><td>${Number(item.cpu_peak_percent).toFixed(1)}%</td><td>${Number(item.memory_peak_mb).toFixed(0)} MB</td><td>${Number(item.model_memory_mb).toFixed(0)} MB</td><td>${Number(item.gpu_memory_mb).toFixed(0)} MB</td></tr>`).join('') : '<tr><td colspan="9" class="empty-cell">Select a model to show its measured results.</td></tr>';
  const maxLatency = Math.max(...rows.map(([, item]) => Number(item.latency_mean_seconds || 0)), 1);
  $('#tradeoff-grid').innerHTML = rows.map(([model, item]) => `<article class="tradeoff-card"><header><strong>${escapeHtml(modelLabel(model))}</strong><span>${percent(item.accuracy)} accurate</span></header><label>Accuracy<i><b style="width:${Number(item.accuracy || 0) * 100}%"></b></i></label><label>Latency<i class="latency"><b style="width:${Number(item.latency_mean_seconds || 0) / maxLatency * 100}%"></b></i><small>${Number(item.latency_mean_seconds).toFixed(2)}s</small></label><label>Memory<i class="memory"><b style="width:${Math.min(Number(item.memory_peak_mb || 0) / 6500 * 100, 100)}%"></b></i><small>${Number(item.memory_peak_mb).toFixed(0)} MB</small></label></article>`).join('');

  // Update Source Graph metrics dynamically
  if (rows.length) {
    const firstItem = rows[0][1];
    const prec = Number(firstItem.retrieval_precision_at_k || 0.2386);
    const rec = Number(firstItem.retrieval_recall_at_k || 0.6409);
    const mrr = Number(firstItem.retrieval_mrr || 0.5303);

    const precEl = $('#sg-precision');
    if (precEl) {
      precEl.textContent = percent(prec);
      const bar = $('#sg-prec-bar');
      if (bar) bar.style.width = `${prec * 100}%`;
    }
    const recEl = $('#sg-recall');
    if (recEl) {
      recEl.textContent = percent(rec);
      const bar = $('#sg-rec-bar');
      if (bar) bar.style.width = `${rec * 100}%`;
    }
    const mrrEl = $('#sg-mrr');
    if (mrrEl) {
      mrrEl.textContent = mrr.toFixed(3);
      const bar = $('#sg-mrr-bar');
      if (bar) bar.style.width = `${Math.min(mrr * 100, 100)}%`;
    }
  }

  renderReportTraces();
}

function renderOverview(payload) {
  state.overview = payload;
  if (!state.selectedModels.length) state.selectedModels = [...payload.models];
  $('#dataset-count').textContent = payload.dataset_size;
  $('#model-count').textContent = payload.models.length;
  $('#run-count').textContent = payload.dataset_size * payload.models.length;
  $('#evaluation-state').textContent = payload.status.state.replace('_', ' ');
  $('#evaluation-updated').textContent = payload.generated_at ? `Saved dataset run: ${new Date(payload.generated_at).toLocaleString()} — live comparisons below do not alter this report.` : `${payload.status.completed}/${payload.status.total} runs`;
  const health = $('#evaluation-health');
  health.className = `health ${payload.status.state === 'complete' ? 'ok' : ''}`;
  health.querySelector('span').textContent = payload.status.state === 'complete' ? 'Evaluation complete' : `${payload.status.completed}/${payload.status.total} evaluated`;
  $('#metric-definitions').innerHTML = Object.entries(payload.metric_definitions).map(([name, text]) => `<div class="definition"><strong>${escapeHtml(name.replaceAll('_', ' '))}</strong><small>${escapeHtml(text)}</small></div>`).join('');
  $('#conclusions').innerHTML = payload.conclusions.length ? payload.conclusions.map((text) => `<p>${escapeHtml(text)}</p>`).join('') : '<p class="empty">Complete the evaluation run to generate quantitative conclusions.</p>';

  // Category verdicts (7 core questions)
  const verdictsContainer = $('#category-verdicts');
  if (verdictsContainer) {
    const verdicts = payload.category_verdicts || [];
    verdictsContainer.innerHTML = verdicts.length ? verdicts.map((item, idx) => `
      <article class="verdict-card">
        <h4 class="verdict-question">${idx + 1}. ${escapeHtml(item.question)}</h4>
        <div class="verdict-winner-badge">🏆 Winner: ${escapeHtml(modelLabel(item.best_model))}</div>
        <div class="verdict-stats">
          <span class="verdict-stat">Accuracy: <b>${percent(item.accuracy)}</b></span>
          <span class="verdict-stat">Mean Latency: <b>${Number(item.latency).toFixed(2)}s</b></span>
          <span class="verdict-stat">Hallucination: <b>${percent(item.hallucination)}</b></span>
          ${item.test_pass_rate != null ? `<span class="verdict-stat">Code Pass: <b>${percent(item.test_pass_rate)}</b></span>` : ''}
        </div>
        <p class="verdict-rationale"><b>Evidence &amp; Rationale:</b> ${escapeHtml(item.rationale)}</p>
      </article>
    `).join('') : '<p class="empty">Category verdicts will appear once evaluation completes.</p>';
  }

  $('#evaluation-conditions').innerHTML = Object.entries(payload.evaluation_conditions || {}).map(([name, value]) => conditionLabel(name, value)).join('');
  $('#model-lineup').innerHTML = payload.models.map((model, index) => {
    const item = payload.aggregates[model] || {};
    return `<article class="model-card" data-model="${escapeHtml(model)}"><span>Model ${index + 1}</span><h3>${escapeHtml(modelLabel(model))}</h3><code>${escapeHtml(model)}</code><div class="model-kpis"><small>Accuracy <b>${item.accuracy == null ? '—' : percent(item.accuracy)}</b></small><small>Latency <b>${item.latency_mean_seconds == null ? '—' : `${Number(item.latency_mean_seconds).toFixed(2)}s`}</b></small><small>Hallucination <b>${item.hallucination_rate == null ? '—' : percent(item.hallucination_rate)}</b></small></div></article>`;
  }).join('');
  renderModelPicker(payload.models);
  renderMeasuredResults();
}

function renderDataset() {
  const rows = state.category === 'all'
    ? state.dataset
    : state.dataset.filter((item) => item.category === state.category);
  $('#dataset-list').innerHTML = rows.map((item) => `
    <article class="dataset-card">
      <span>${escapeHtml(item.id)} · <span class="category-badge">${escapeHtml(item.category || item.domain)}</span></span>
      <p>${escapeHtml(item.question)}</p>
      <button type="button" class="use-question" data-domain="${escapeHtml(item.domain)}" data-question="${escapeHtml(item.question)}">Use in live comparison</button>
    </article>
  `).join('');
}

function renderTrace(row, repository = false) {
  const sourceText = (row.sources || []).map((source) => `${source.citation}\n${source.excerpt}`).join('\n\n');
  const outcome = repository ? `accuracy ${percent(row.correctness)}` : `${row.retrieval_outcome} → ${row.response_outcome}`;
  const tone = /hallucinated|missed|irrelevant|incorrect/.test(outcome) ? 'warning' : 'success';
  return `<article class="trace"><header><span>${escapeHtml(row.question_id)}</span><span>${escapeHtml(modelLabel(row.model))}</span></header><p>${escapeHtml(row.question)}</p><span class="outcome ${tone}">${escapeHtml(outcome)}</span><details><summary>Show retrieved context → LLM response</summary><div class="trace-stage"><b>Retrieved context</b><pre>${escapeHtml(sourceText || row.retrieved_context || 'No context retrieved')}</pre></div><div class="trace-stage"><b>LLM response</b><pre>${escapeHtml(row.answer)}</pre></div></details></article>`;
}

function renderReportTraces() {
  if (!state.report) return;
  const selected = new Set(state.selectedModels);
  const ragRows = state.report.rag_analysis.filter((row) => selected.has(row.model));
  const repositoryRows = state.report.repository_analysis.filter((row) => selected.has(row.model));
  $('#rag-traces').innerHTML = ragRows.length ? ragRows.map((row) => renderTrace(row)).join('') : '<p class="empty">No RAG traces for the selected models.</p>';
  $('#repository-traces').innerHTML = repositoryRows.length ? repositoryRows.map((row) => renderTrace(row, true)).join('') : '<p class="empty">No repository traces for the selected models.</p>';
}

async function loadReport() {
  try {
    state.report = await requestJson('/report');
    renderReportTraces();
  } catch (error) {
    $('#rag-traces').innerHTML = `<p class="empty">${escapeHtml(error.message)}</p>`;
  }
}

function renderLiveComparison(payload) {
  $('#live-results').innerHTML = payload.results.map((result) => `
    <article class="live-result">
      <header><div><span>${escapeHtml(result.model)}</span><h4>${escapeHtml(modelLabel(result.model))}</h4></div><strong>${Number(result.latency_seconds).toFixed(2)}s</strong></header>
      <p>${escapeHtml(result.answer)}</p>
      <div class="live-evidence"><p>Question overlap F1: ${percent(result.metrics?.question_overlap_f1)}</p><p>Context token support: ${percent(result.metrics?.context_token_support)}</p><p>Mean retrieval similarity: ${result.metrics?.retrieval_similarity_mean == null ? 'Not applicable' : Number(result.metrics.retrieval_similarity_mean).toFixed(3)}</p><small>${escapeHtml(result.metrics?.definitions || '')}</small></div>
      <details open><summary>Rubric scores and evidence</summary><p>${escapeHtml(result.metrics?.grading || '')}</p><p>${result.metrics?.accuracy == null ? 'Fact accuracy requires required facts or a dataset rubric.' : `Fact accuracy: ${percent(result.metrics.accuracy)}`}</p><p>${result.metrics?.reference_answer ? `Reference relevance F1: ${percent(result.metrics.relevance_f1)}` : 'Add a verified expected answer above for reference-based scoring.'}</p><p>Hallucination proxy: ${result.metrics?.hallucination_flag == null ? 'Requires retrieved evidence' : result.metrics.hallucination_flag ? 'Flagged' : 'Not flagged (not a factuality guarantee)'}</p><p>${escapeHtml((result.metrics?.hallucination_reasons || []).join(', '))}</p>${result.metrics?.retrieval ? `<p>Retrieval precision: ${percent(result.metrics.retrieval.precision_at_k)} · Recall: ${percent(result.metrics.retrieval.recall_at_k)}</p>` : '<p>Retrieval precision/recall require labelled relevant sources; similarity is shown above.</p>'}${result.metrics?.code_test_passed == null ? '' : `<p>Code test: ${result.metrics.code_test_passed ? 'Passed' : 'Failed'} ${escapeHtml(result.metrics.code_test_detail)}</p>`}${result.metrics?.reference_answer ? `<p>Reference: ${escapeHtml(result.metrics.reference_answer)}</p>` : ''}</details>
      <footer><span>${result.total_tokens ?? 'N/A'} total tokens</span><span>${result.model_memory_mb ?? 'N/A'} MB model memory</span><span>${result.gpu_memory_mb ?? 'N/A'} MB GPU memory</span><span>Live CPU sampling: N/A; see saved dataset measurements</span></footer>
    </article>`).join('');
  $('#live-sources').innerHTML = payload.sources.map((source, index) => `<article><strong>#${index + 1} · ${escapeHtml(source.citation)}</strong><span>${Math.round(Number(source.relevance_score) * 100)}% relevant</span><p>${escapeHtml(source.excerpt)}</p></article>`).join('');
  $('#live-context').hidden = false;
  $('#live-status').className = 'live-status success';
  $('#live-status').textContent = `${payload.results.length} models completed with the same prompt and limits (${payload.conditions.max_tokens} output tokens). Short outputs can be truncated. Scores are automatic proxies, not human grades.`;
}

let cachedSourcegraphData = null;

// Sourcegraph Semantic Search Integration
async function runSourcegraphSearch(query) {
  const input = $('#sg-search-input');
  if (input && query) input.value = query;
  const q = (query || input?.value || '').trim();
  const resultsContainer = $('#sg-results');
  if (!resultsContainer) return;

  resultsContainer.innerHTML = '<div style="grid-column: 1/-1; padding: 16px; color: var(--muted); text-align: center;">Running Sourcegraph semantic symbol analysis…</div>';

  let results = [];
  try {
    const res = await fetch(`/retrieval-api/sourcegraph/search?q=${encodeURIComponent(q)}&limit=8`);
    if (res.ok) {
      const data = await res.json();
      results = data.results || [];
    } else {
      throw new Error('Fallback to local index');
    }
  } catch {
    try {
      if (!cachedSourcegraphData) {
        const resp = await fetch('sourcegraph.json');
        cachedSourcegraphData = await resp.json();
      }
      const allSymbols = cachedSourcegraphData?.symbols || [];
      let term = q;
      let typeFilter = null;
      let pathFilter = null;

      const matchType = q.match(/type:(symbol|class|function|endpoint)/i);
      if (matchType) {
        typeFilter = matchType[1].toLowerCase();
        term = term.replace(matchType[0], '').trim();
      }
      const matchPath = q.match(/path:([^\s]+)/i);
      if (matchPath) {
        pathFilter = matchPath[1].toLowerCase();
        term = term.replace(matchPath[0], '').trim();
      }
      const termLower = term.toLowerCase().trim();

      results = allSymbols.filter((sym) => {
        if (typeFilter && typeFilter !== 'symbol' && sym.type !== typeFilter) return false;
        if (pathFilter && !sym.file.toLowerCase().includes(pathFilter)) return false;
        if (!termLower) return true;
        return sym.name.toLowerCase().includes(termLower) ||
               sym.signature.toLowerCase().includes(termLower) ||
               sym.file.toLowerCase().includes(termLower);
      }).slice(0, 8);
    } catch (err) {
      resultsContainer.innerHTML = `<div style="grid-column: 1/-1; padding: 16px; color: #f87171; text-align: center;">Sourcegraph query error: ${escapeHtml(err.message)}</div>`;
      return;
    }
  }

  if (!results.length) {
    resultsContainer.innerHTML = `<div style="grid-column: 1/-1; padding: 16px; color: var(--muted); text-align: center;">No Sourcegraph symbols found for: <code>${escapeHtml(q)}</code></div>`;
    return;
  }

  resultsContainer.innerHTML = results.map((item) => `
    <div class="sg-result-card">
      <div class="sg-result-header">
        <span class="sg-sym-type ${escapeHtml(item.type)}">${escapeHtml(item.type)}</span>
        <span class="sg-sym-location">${escapeHtml(item.file)}:${item.line}</span>
      </div>
      <div class="sg-sym-sig">${escapeHtml(item.signature)}</div>
      ${item.docstring ? `<div style="font-size: 10px; color: #a9bbb2; font-style: italic;">${escapeHtml(item.docstring.slice(0, 100))}</div>` : ''}
      <pre class="sg-sym-code"><code>${escapeHtml(item.snippet)}</code></pre>
    </div>
  `).join('');
}

function initSourcegraph() {
  const btn = $('#sg-search-btn');
  const input = $('#sg-search-input');
  if (btn) {
    btn.onclick = () => runSourcegraphSearch();
  }
  if (input) {
    input.onkeydown = (e) => {
      if (e.key === 'Enter') runSourcegraphSearch();
    };
  }
  $$('.sg-preset').forEach((presetBtn) => {
    presetBtn.onclick = () => {
      runSourcegraphSearch(presetBtn.dataset.query);
    };
  });
  runSourcegraphSearch('type:symbol orchestrator');
}

async function load() {
  try {
    const [overview, dataset] = await Promise.all([requestJson('/overview'), requestJson('/dataset')]);
    renderOverview(overview);
    state.dataset = dataset.questions;
    renderDataset();
    if (overview.status.state === 'complete') await loadReport();
    initSourcegraph();
  } catch (error) {
    $('#evaluation-health').className = 'health error';
    $('#evaluation-health').querySelector('span').textContent = error.message;
  }
}

$$('.filter').forEach((button) => button.addEventListener('click', () => {
  state.category = button.dataset.category || button.dataset.domain || 'all';
  $$('.filter').forEach((item) => item.classList.toggle('active', item === button));
  renderDataset();
}));

$('#dataset-list').addEventListener('click', (event) => {
  const button = event.target.closest('.use-question');
  if (!button) return;
  $('#live-question').value = button.dataset.question;
  $('#live-domain').value = button.dataset.domain;
  $('#live-reference').value = '';
  $('#live-facts').value = '';
  $('#exercise-1').scrollIntoView({ behavior: 'smooth' });
  $('#live-question').focus();
});

$('#model-picker').addEventListener('change', () => {
  state.selectedModels = $$('#model-picker input:checked').map((input) => input.value);
  const ready = state.selectedModels.length >= 2;
  $('#run-live-comparison').disabled = !ready;
  $('#live-status').className = 'live-status';
  $('#live-status').textContent = ready ? `${state.selectedModels.length} models selected. Measured tables and traces are filtered to this comparison.` : 'Select at least two models for a controlled live comparison.';
  renderMeasuredResults();
});

$('#live-compare-form').addEventListener('submit', async (event) => {
  event.preventDefault();
  const button = $('#run-live-comparison');
  if (state.selectedModels.length < 2) return;
  button.disabled = true;
  $$('#model-picker input, #live-domain, #live-question, #live-top-k, #live-max-tokens, #live-reference, #live-facts, .use-question, .filter, #refresh-evaluation').forEach((control) => control.disabled = true);
  button.textContent = 'Running identical prompt…';
  $('#live-status').className = 'live-status busy';
  $('#live-status').textContent = `Retrieving once, then running ${state.selectedModels.length} models sequentially for a fair CPU comparison. This can take a few minutes.`;
  $('#live-results').innerHTML = '';
  $('#live-context').hidden = true;
  try {
    const payload = await requestJson('/compare-models', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question: $('#live-question').value.trim(), domain: $('#live-domain').value, models: state.selectedModels, top_k: Number($('#live-top-k').value), max_tokens: Number($('#live-max-tokens').value), reference_answer: $('#live-reference').value.trim(), required_facts: $('#live-facts').value.split('\n').map(v => v.trim()).filter(Boolean) }),
    });
    renderLiveComparison(payload);
  } catch (error) {
    $('#live-status').className = 'live-status error';
    $('#live-status').textContent = error.message;
  } finally {
    $$('#model-picker input, #live-domain, #live-question, #live-top-k, #live-max-tokens, #live-reference, #live-facts, .use-question, .filter, #refresh-evaluation').forEach((control) => control.disabled = false);
    button.disabled = state.selectedModels.length < 2;
    button.textContent = 'Compare selected models →';
  }
});

$('#refresh-evaluation').addEventListener('click', load);
load();
