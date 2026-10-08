const page = document.body.dataset.ui;
const health = document.querySelector('#service-health');
const escapeHtml = (value) => { const node = document.createElement('div'); node.textContent = String(value ?? ''); return node.innerHTML; };

async function requestJson(path, options = {}) {
  let response;
  try { response = await fetch(path, options); }
  catch { throw new Error('Cannot connect to the NoteBuddy backend'); }
  let payload = {};
  try { payload = await response.json(); } catch { /* response may be empty */ }
  if (!response.ok) throw new Error(payload.detail || `Request failed with HTTP ${response.status}`);
  return payload;
}

function setHealth(ok, text) {
  health.className = `health ${ok ? 'ok' : 'error'}`;
  health.querySelector('span').textContent = text;
}

function formatBytes(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1048576) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / 1048576).toFixed(1)} MB`;
}

async function loadKnowledge() {
  const payload = await requestJson('/data-api/documents');
  document.querySelector('#metric-documents').textContent = payload.document_count;
  document.querySelector('#metric-chunks').textContent = payload.collection_total;
  const list = document.querySelector('#document-list');
  if (!payload.documents.length) {
    list.innerHTML = '<p class="empty">No course documents yet.</p>';
    return;
  }
  list.innerHTML = payload.documents.map((item) => `<article class="document"><span>${escapeHtml(item.extension.slice(1).toUpperCase())}</span><div><strong title="${escapeHtml(item.name)}">${escapeHtml(item.name)}</strong><small>${formatBytes(item.size_bytes)} · indexed</small></div><button class="delete-document" type="button" data-filename="${escapeHtml(item.name)}" aria-label="Remove ${escapeHtml(item.name)}">Remove</button></article>`).join('');
}

async function initRetrieval() {
  try {
    const status = await requestJson('/retrieval-api/health');
    setHealth(true, `Retrieval online · ${status.details.chunks} chunks`);
    await loadKnowledge();
  } catch (error) { setHealth(false, error.message); }

  document.querySelector('#refresh-knowledge').addEventListener('click', () => loadKnowledge().catch(showKnowledgeError));
  document.querySelector('#rebuild-index').addEventListener('click', async (event) => {
    const button = event.currentTarget; button.disabled = true; button.textContent = 'Building embeddings…';
    try {
      const report = await requestJson('/data-api/ingest', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({reset:true})});
      showKnowledgeFeedback(`${report.documents} documents indexed into ${report.chunks} chunks.`, true);
      await loadKnowledge();
    } catch (error) { showKnowledgeError(error); }
    finally { button.disabled = false; button.textContent = 'Rebuild vector index'; }
  });

  document.querySelector('#knowledge-files').addEventListener('change', async (event) => {
    const files = [...event.target.files]; if (!files.length) return;
    const body = new FormData(); files.forEach((file) => body.append('files', file));
    showKnowledgeFeedback(`Uploading ${files.length} file(s)…`);
    try {
      const upload = await requestJson('/data-api/documents/upload', {method:'POST', body});
      if (!upload.uploaded.length) throw new Error(upload.rejected.join('; ') || 'No files accepted');
      const report = await requestJson('/data-api/ingest', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({reset:true})});
      showKnowledgeFeedback(`${upload.uploaded.length} uploaded; ${report.chunks} chunks ready.`, true);
      await loadKnowledge();
    } catch (error) { showKnowledgeError(error); }
    finally { event.target.value = ''; }
  });

  document.querySelector('#document-list').addEventListener('click', async (event) => {
    const button = event.target.closest('.delete-document');
    if (!button) return;
    const filename = button.dataset.filename;
    if (!window.confirm(`Remove “${filename}” from document storage and the vector index?`)) return;
    button.disabled = true;
    showKnowledgeFeedback(`Removing ${filename} and its indexed chunks…`);
    try {
      const result = await requestJson(`/data-api/documents/${encodeURIComponent(filename)}`, {method:'DELETE'});
      showKnowledgeFeedback(`${result.deleted.name} removed. ${result.remaining_documents} document(s) and ${result.collection_total} chunks remain.`, true);
      await loadKnowledge();
    } catch (error) { showKnowledgeError(error); button.disabled = false; }
  });

  document.querySelector('#retrieval-form').addEventListener('submit', async (event) => {
    event.preventDefault(); const form = event.currentTarget; const button = form.querySelector('button');
    button.disabled = true; button.textContent = 'Embedding + searching…';
    const results = document.querySelector('#retrieval-results'); results.classList.add('busy');
    try {
      const payload = await requestJson('/retrieval-api/retrieve', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({question:document.querySelector('#retrieval-question').value.trim(), top_k:Number(document.querySelector('#top-k').value)})});
      results.innerHTML = payload.sources.length ? payload.sources.map((source, index) => `<article class="result-card"><header><span>#${index + 1} · ${escapeHtml(source.citation)}</span><span>${Math.round(source.relevance_score * 100)}% relevant</span></header><p>${escapeHtml(source.excerpt)}</p></article>`).join('') : '<div class="empty-state"><strong>No evidence found</strong><p>Add and index course material before searching.</p></div>';
    } catch (error) { results.innerHTML = `<div class="empty-state"><strong>Search failed</strong><p>${escapeHtml(error.message)}</p></div>`; }
    finally { results.classList.remove('busy'); button.disabled = false; button.textContent = 'Search knowledge base →'; }
  });
}

function showKnowledgeFeedback(message, success = false) { const node = document.querySelector('#knowledge-feedback'); node.textContent = message; node.className = `feedback ${success ? 'success' : ''}`; }
function showKnowledgeError(error) { const node = document.querySelector('#knowledge-feedback'); node.textContent = error.message; node.className = 'feedback error'; }

async function initLlm() {
  const modelSelect = document.querySelector('#model-select');
  try {
    const [status, catalog] = await Promise.all([requestJson('/llm-api/health'), requestJson('/llm-api/models')]);
    setHealth(true, `LLM online · Ollama ${status.details.version}`);
    modelSelect.innerHTML = catalog.models.map((model) => `<option value="${escapeHtml(model.id)}" ${model.id === catalog.default_model ? 'selected' : ''} ${model.available ? '' : 'disabled'}>${escapeHtml(model.label)}${model.available ? '' : ' · unavailable'}</option>`).join('');
    document.querySelector('#model-availability').textContent = `${catalog.models.filter((model) => model.available).length}/${catalog.models.length} models ready in Ollama`;
  } catch (error) { setHealth(false, error.message); }

  const output = document.querySelector('#llm-output');
  document.querySelector('#clear-output').addEventListener('click', () => { output.innerHTML = '<div class="empty-state"><span>✦</span><strong>Ready for a prompt</strong><p>The response will appear here with the model name and request timing.</p></div>'; });
  document.querySelector('#llm-form').addEventListener('submit', async (event) => {
    event.preventDefault(); const button = event.currentTarget.querySelector('button'); const started = performance.now();
    const selectedLabel = modelSelect.options[modelSelect.selectedIndex]?.textContent || modelSelect.value;
    button.disabled = true; button.textContent = `${selectedLabel} is thinking…`; output.classList.add('busy');
    try {
      const payload = await requestJson('/llm-api/generate', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({prompt:document.querySelector('#llm-prompt').value.trim(), model:modelSelect.value})});
      const seconds = ((performance.now() - started) / 1000).toFixed(1);
      const tokens = Number(payload.prompt_tokens || 0) + Number(payload.completion_tokens || 0);
      output.innerHTML = `<div class="output-meta"><span>Model · ${escapeHtml(payload.model)}</span><span>${seconds}s · ${tokens || '—'} tokens · no context</span></div><div class="output-text">${escapeHtml(payload.answer)}</div>`;
    } catch (error) { output.innerHTML = `<div class="empty-state"><strong>Generation failed</strong><p>${escapeHtml(error.message)}</p></div>`; }
    finally { output.classList.remove('busy'); button.disabled = false; button.textContent = 'Generate response →'; }
  });
}

if (page === 'retrieval') initRetrieval();
if (page === 'llm') initLlm();
