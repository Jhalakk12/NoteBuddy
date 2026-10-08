// The browser talks only to the website origin. Nginx privately routes /api
// to the Application Service; Retrieval, LLM, Data, Chroma, and Ollama never
// need public browser-facing ports.
const API_BASE = window.NOTEBUDDY_API || "/api";

const state = { mode: "rag", busy: false, knowledge: null };
const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => [...document.querySelectorAll(selector)];

const menuButton = $(".menu-button");
const mobileNav = $(".mobile-nav");
const form = $("#question-form");
const input = $("#question-input");
const messages = $("#messages");
const sendButton = form?.querySelector("button[type='submit']");
const clearButton = $("#clear-chat");
const uploadButton = $("#upload-button");
const mobileUploadButton = $("#mobile-upload-button");
const documentInput = $("#document-input");
const uploadStatus = $("#upload-status");
const connectionButton = $("#connection-state");
const systemPopover = $("#system-popover");

menuButton?.addEventListener("click", () => {
  const isOpen = mobileNav.classList.toggle("open");
  menuButton.setAttribute("aria-expanded", String(isOpen));
});

$$('.mobile-nav a').forEach((link) => {
  link.addEventListener('click', () => {
    mobileNav.classList.remove('open');
    menuButton?.setAttribute('aria-expanded', 'false');
  });
});

$$('.faq-item button').forEach((button) => {
  button.addEventListener('click', () => {
    const item = button.closest('.faq-item');
    const isOpen = item.classList.toggle('open');
    button.setAttribute('aria-expanded', String(isOpen));
  });
});

const revealObserver = new IntersectionObserver(
  (entries) => entries.forEach((entry) => {
    if (entry.isIntersecting) {
      entry.target.classList.add('visible');
      revealObserver.unobserve(entry.target);
    }
  }),
  { threshold: 0.08 },
);
$$('.reveal').forEach((element) => revealObserver.observe(element));

function escapeHtml(value) {
  const div = document.createElement('div');
  div.textContent = String(value ?? '');
  return div.innerHTML;
}

async function api(path, options = {}) {
  let response;
  try {
    response = await fetch(`${API_BASE}${path}`, options);
  } catch {
    throw new Error('Cannot connect to the NoteBuddy backend');
  }
  let payload;
  try { payload = await response.json(); } catch { payload = {}; }
  if (!response.ok) throw new Error(payload.detail || `Request failed with HTTP ${response.status}`);
  return payload;
}

function citationMarkup(source) {
  return `<button class="citation-chip" type="button" data-citation="${escapeHtml(source.citation)}" data-excerpt="${escapeHtml(source.excerpt || '')}" data-score="${escapeHtml(source.relevance_score ?? '')}"><span>↗</span> ${escapeHtml(source.citation)}</button>`;
}

function wireCitations(container) {
  container.querySelectorAll('.citation-chip[data-citation]').forEach((button) => {
    button.addEventListener('click', () => {
      container.querySelector('.source-preview')?.remove();
      const preview = document.createElement('div');
      preview.className = 'source-preview';
      const score = button.dataset.score ? ` · relevance ${Math.round(Number(button.dataset.score) * 100)}%` : '';
      preview.innerHTML = `<strong>${escapeHtml(button.dataset.citation)}</strong><p>${escapeHtml(button.dataset.excerpt || 'No excerpt available.')}</p><small>Retrieved course evidence${score}</small>`;
      button.parentElement.append(preview);
    });
  });
}

function appendMessage(role, content, sources = [], orchestration = []) {
  const article = document.createElement('article');
  article.className = `message ${role === 'user' ? 'user-message' : 'assistant-message'}`;
  article.innerHTML = `
    <div class="avatar ${role === 'user' ? 'user-avatar' : 'assistant-avatar'}">${role === 'user' ? 'You' : 'N'}</div>
    <div class="message-content">
      <span class="message-name">${role === 'user' ? 'You' : 'NoteBuddy'}</span>
      <p>${escapeHtml(content)}</p>
      ${sources.slice(0, 4).map(citationMarkup).join('')}
      ${orchestration.length ? `<div class="orchestration-trace">${orchestration.map(escapeHtml).join('<span>→</span>')}</div>` : ''}
    </div>`;
  messages.append(article);
  wireCitations(article);
  messages.scrollTop = messages.scrollHeight;
  return article;
}

function appendTyping(label = 'Retrieving course context and generating an answer') {
  const article = document.createElement('article');
  article.className = 'message assistant-message';
  article.innerHTML = `<div class="avatar assistant-avatar">N</div><div class="message-content"><span class="message-name">NoteBuddy · ${escapeHtml(label)}</span><div class="typing" aria-label="Working"><i></i><i></i><i></i></div></div>`;
  messages.append(article);
  messages.scrollTop = messages.scrollHeight;
  return article;
}

function appendComparison(payload) {
  const article = document.createElement('article');
  article.className = 'comparison-message';
  const sources = payload.with_rag.sources || [];
  article.innerHTML = `
    <p class="comparison-label">Same question · two answer paths</p>
    <p class="comparison-question">${escapeHtml(payload.question)}</p>
    <div class="comparison-grid-live">
      <div class="answer-variant"><small>Without retrieved context</small><p>${escapeHtml(payload.without_rag.answer)}</p><span class="variant-status">Base Code Llama</span></div>
      <div class="answer-variant grounded"><small>With NoteBuddy RAG</small><p>${escapeHtml(payload.with_rag.answer)}</p><span class="variant-status">Grounded + cited</span>${sources.slice(0, 2).map(citationMarkup).join('')}<div class="orchestration-trace">${(payload.with_rag.orchestration || []).map(escapeHtml).join('<span>→</span>')}</div></div>
    </div>`;
  messages.append(article);
  wireCitations(article);
  messages.scrollTop = messages.scrollHeight;
}

function setBusy(busy) {
  state.busy = busy;
  sendButton.disabled = busy;
  uploadButton.disabled = busy;
  mobileUploadButton.disabled = busy;
}

function setMode(mode) {
  state.mode = mode;
  $$('.mode-switch button').forEach((button) => button.classList.toggle('active', button.dataset.mode === mode));
  if (mode === 'compare') {
    $('#mode-overline').textContent = 'RAG impact demonstration';
    $('#mode-heading').textContent = 'Compare both answer paths';
    $('#composer-mode').innerHTML = '<span>⇄</span> RAG vs. no RAG';
    input.placeholder = 'Ask one question to compare both answers…';
  } else {
    $('#mode-overline').textContent = 'Grounded conversation';
    $('#mode-heading').textContent = 'Ask your course';
    $('#composer-mode').innerHTML = '<span>✦</span> RAG + citations';
    input.placeholder = 'Ask about a deadline, topic, lecture, or assessment…';
  }
}

$$('.mode-switch button').forEach((button) => button.addEventListener('click', () => setMode(button.dataset.mode)));

input?.addEventListener('input', () => {
  input.style.height = 'auto';
  input.style.height = `${Math.min(input.scrollHeight, 110)}px`;
});

input?.addEventListener('keydown', (event) => {
  if (event.key === 'Enter' && !event.shiftKey) {
    event.preventDefault();
    form?.requestSubmit();
  }
});

form?.addEventListener('submit', async (event) => {
  event.preventDefault();
  const question = input.value.trim();
  if (!question || state.busy) return;
  appendMessage('user', question);
  input.value = '';
  input.style.height = 'auto';
  setBusy(true);
  const typing = appendTyping(state.mode === 'compare' ? 'Running both model paths' : undefined);
  try {
    const payload = await api(state.mode === 'compare' ? '/compare' : '/ask', {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ question }),
    });
    typing.remove();
    if (state.mode === 'compare') appendComparison(payload);
    else appendMessage('assistant', payload.answer, payload.sources || [], payload.orchestration || []);
  } catch (error) {
    typing.remove();
    appendMessage('assistant', `${error.message}. Open System status for details or ask the administrator to check the deployment.`);
  } finally {
    setBusy(false);
    input.focus();
  }
});

clearButton?.addEventListener('click', () => {
  messages.innerHTML = '';
  appendMessage('assistant', 'Conversation cleared. What would you like to find in your course materials?');
});

function fileIcon(extension) {
  if (extension === '.pdf') return 'P';
  if (extension === '.md') return 'M';
  return 'T';
}

function formatBytes(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function renderKnowledge(payload) {
  state.knowledge = payload;
  const fileLabel = payload.document_count === 1 ? 'file' : 'files';
  $('#document-count').textContent = `${payload.document_count} ${fileLabel}`;
  $('#index-title').textContent = payload.collection_total ? 'Knowledge ready' : 'Knowledge not indexed';
  $('#index-detail').textContent = `${payload.collection_total} Chroma ${payload.collection_total === 1 ? 'chunk' : 'chunks'}`;
  $('#mobile-index-status').textContent = payload.collection_total ? `${payload.document_count} ${fileLabel} · ${payload.collection_total} chunks ready` : payload.document_count ? 'Files uploaded · indexing needed' : 'Add course materials to begin';
  const list = $('#source-list');
  if (!payload.documents.length) {
    list.innerHTML = '<div class="knowledge-empty"><span>⇧</span><strong>No documents yet</strong><small>Upload PDF, TXT, or Markdown files</small></div>';
    return;
  }
  list.innerHTML = payload.documents.map((document) => `
    <div class="source-item"><span class="source-icon">${fileIcon(document.extension)}</span><span><strong title="${escapeHtml(document.name)}">${escapeHtml(document.name)}</strong><small>${formatBytes(document.size_bytes)} · ${escapeHtml(document.extension.slice(1).toUpperCase())}</small></span><span class="source-check">${payload.collection_total ? '✓' : '○'}</span></div>`).join('');
}

async function loadKnowledge() {
  try { renderKnowledge(await api('/knowledge/status')); }
  catch (error) {
    uploadStatus.textContent = error.message;
    uploadStatus.className = 'upload-status error';
    $('#mobile-index-status').textContent = 'Knowledge service unavailable';
  }
}

function chooseFiles() { if (!state.busy) documentInput.click(); }
uploadButton?.addEventListener('click', chooseFiles);
mobileUploadButton?.addEventListener('click', chooseFiles);

documentInput?.addEventListener('change', async () => {
  const files = [...documentInput.files];
  if (!files.length) return;
  setBusy(true);
  uploadStatus.className = 'upload-status';
  uploadStatus.textContent = `Uploading ${files.length} ${files.length === 1 ? 'file' : 'files'}…`;
  try {
    const formData = new FormData();
    files.forEach((file) => formData.append('files', file));
    const upload = await api('/knowledge/upload', { method: 'POST', body: formData });
    if (!upload.uploaded.length) throw new Error(upload.rejected.join('; ') || 'No supported files were uploaded');
    uploadStatus.textContent = 'Files uploaded. Creating embeddings and rebuilding the index…';
    const report = await api('/knowledge/ingest', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ reset: true }) });
    const rejected = upload.rejected.length ? ` ${upload.rejected.length} rejected.` : '';
    uploadStatus.textContent = `${report.documents} documents indexed into ${report.chunks} chunks.${rejected}`;
    uploadStatus.className = 'upload-status success';
    await loadKnowledge();
    await loadSystemStatus();
    appendMessage('assistant', `Knowledge base updated: ${report.documents} documents and ${report.chunks} cited chunks are ready. Ask me a question about them.`);
  } catch (error) {
    uploadStatus.textContent = error.message;
    uploadStatus.className = 'upload-status error';
  } finally {
    documentInput.value = '';
    setBusy(false);
  }
});

function renderSystemStatus(payload) {
  const ok = payload.overall === 'ok';
  connectionButton.classList.toggle('degraded', !ok);
  connectionButton.querySelector('b').textContent = ok ? 'Four services online' : 'System needs attention';
  connectionButton.setAttribute('aria-label', ok ? 'All NoteBuddy services online' : 'Some NoteBuddy services unavailable');
  $('#hero-system-status').innerHTML = `<span class="status-dot"></span>${ok ? 'All NoteBuddy systems online' : 'NoteBuddy is partially unavailable'}`;
  $('#service-status-list').innerHTML = payload.services.map((service) => `
    <div class="service-status-row ${service.status === 'ok' ? '' : 'unavailable'}"><i></i><div><strong>${escapeHtml(service.name)}</strong><small title="${escapeHtml(service.detail || service.url)}">${escapeHtml(service.detail || service.url)}</small></div></div>`).join('');
}

async function loadSystemStatus() {
  try { renderSystemStatus(await api('/system/status')); }
  catch (error) {
    renderSystemStatus({ overall: 'degraded', services: [{ name: 'application', status: 'unavailable', url: API_BASE, detail: error.message }] });
  }
}

connectionButton?.addEventListener('click', () => {
  const willOpen = systemPopover.hidden;
  systemPopover.hidden = !willOpen;
  connectionButton.setAttribute('aria-expanded', String(willOpen));
  if (willOpen) loadSystemStatus();
});
$('#refresh-status')?.addEventListener('click', loadSystemStatus);
document.addEventListener('click', (event) => {
  if (!systemPopover.hidden && !systemPopover.contains(event.target) && !connectionButton.contains(event.target)) {
    systemPopover.hidden = true;
    connectionButton.setAttribute('aria-expanded', 'false');
  }
});

// Interactive Prompt Chips: 1-click ask
document.addEventListener('click', (event) => {
  const chip = event.target.closest('.prompt-chip');
  if (!chip || state.busy) return;
  const promptText = chip.dataset.prompt || chip.textContent.trim().replace(/^[^\w]+/, '');
  if (input) {
    input.value = promptText;
    input.style.height = 'auto';
    input.style.height = `${Math.min(input.scrollHeight, 110)}px`;
    form?.requestSubmit();
  }
});

// Architecture Modal Toggle
const archModal = $('#arch-modal');
$('#open-arch-modal')?.addEventListener('click', () => {
  if (archModal) archModal.hidden = false;
});
$('#close-arch-modal')?.addEventListener('click', () => {
  if (archModal) archModal.hidden = true;
});
archModal?.addEventListener('click', (e) => {
  if (e.target === archModal) archModal.hidden = true;
});

setMode('rag');
Promise.all([loadSystemStatus(), loadKnowledge()]);

