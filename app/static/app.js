/**
 * Agentic RAG System - Frontend Application Logic
 */

let conversationHistory = [];
let selectedFile = null;

document.addEventListener("DOMContentLoaded", () => {
  fetchStatus();
  initDropzone();
  // Poll status every 30 seconds
  setInterval(fetchStatus, 30000);
});

// --- Tab Switching ---
function switchTab(tabId) {
  // Update Buttons
  document.getElementById("tabChatBtn").classList.toggle("active", tabId === "chat");
  document.getElementById("tabSearchBtn").classList.toggle("active", tabId === "search");
  document.getElementById("tabIngestBtn").classList.toggle("active", tabId === "ingest");

  // Update Content Sections
  document.getElementById("chatSection").classList.toggle("active", tabId === "chat");
  document.getElementById("searchSection").classList.toggle("active", tabId === "search");
  document.getElementById("ingestSection").classList.toggle("active", tabId === "ingest");

  if (tabId === "chat") {
    document.getElementById("chatInput").focus();
  } else if (tabId === "search") {
    document.getElementById("searchInput").focus();
  }
}

// --- Fetch System & Vector DB Status ---
async function fetchStatus() {
  const statusPill = document.getElementById("statusPill");
  const statusDot = statusPill.querySelector(".status-dot");
  const statusText = document.getElementById("statusText");
  const vectorCount = document.getElementById("vectorCount");

  try {
    const res = await fetch("/api/status");
    if (!res.ok) throw new Error("Status endpoint error");
    const data = await res.json();

    const count = data.collection ? data.collection.points_count : 0;
    vectorCount.innerText = Number(count).toLocaleString();

    statusDot.className = "status-dot active";
    statusText.innerText = "Qdrant Online";

    // Update badges if on ingestion tab
    if (document.getElementById("collectionNameBadge")) {
      document.getElementById("collectionNameBadge").innerText = data.collection.collection_name || "enterprise_rag";
    }
    if (document.getElementById("embeddingModelBadge")) {
      document.getElementById("embeddingModelBadge").innerText = `${data.embedding_model} (${data.embedding_dim}-dim)`;
    }
    if (document.getElementById("llmModelBadge")) {
      document.getElementById("llmModelBadge").innerText = data.llm_model || "openai/gpt-oss-120b";
    }
  } catch (err) {
    statusDot.className = "status-dot error";
    statusText.innerText = "DB Disconnected";
    console.error("Status check failed:", err);
  }
}

// --- 1. CHAT TAB LOGIC ---
async function handleChatSubmit(event) {
  event.preventDefault();
  const input = document.getElementById("chatInput");
  const sendBtn = document.getElementById("chatSendBtn");
  const message = input.value.trim();
  const topK = parseInt(document.getElementById("chatTopK").value, 10) || 5;

  if (!message) return;

  // Append user message to UI
  appendMessage("user", message);
  input.value = "";
  sendBtn.disabled = true;

  // Add typing indicator / assistant placeholder
  const loadingMessageId = appendLoadingMessage();

  try {
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        message: message,
        history: conversationHistory,
        top_k: topK
      })
    });

    if (!response.ok) {
      const err = await response.json();
      throw new Error(err.detail || "Failed to fetch response");
    }

    const data = await response.json();

    // Remove loading placeholder
    removeLoadingMessage(loadingMessageId);

    // Append Assistant Answer with Sources
    appendAssistantMessage(data.answer, data.sources, data.model_used);

    // Save to conversation history
    conversationHistory.push({ role: "user", content: message });
    conversationHistory.push({ role: "assistant", content: data.answer });

  } catch (error) {
    removeLoadingMessage(loadingMessageId);
    appendAssistantMessage(`❌ **Error:** ${error.message}`, [], "error");
  } finally {
    sendBtn.disabled = false;
    input.focus();
  }
}

function appendMessage(role, text) {
  const container = document.getElementById("chatMessages");
  const msgDiv = document.createElement("div");
  msgDiv.className = `message ${role}-message`;

  const bubble = document.createElement("div");
  bubble.className = "message-bubble";
  bubble.innerText = text;

  msgDiv.appendChild(bubble);
  container.appendChild(msgDiv);
  container.scrollTop = container.scrollHeight;
}

function appendLoadingMessage() {
  const container = document.getElementById("chatMessages");
  const id = "loading-" + Date.now();
  const msgDiv = document.createElement("div");
  msgDiv.id = id;
  msgDiv.className = "message assistant-message";

  const bubble = document.createElement("div");
  bubble.className = "message-bubble";
  bubble.innerHTML = '<span class="spinner"></span> <em>Searching knowledge base & reasoning...</em>';

  msgDiv.appendChild(bubble);
  container.appendChild(msgDiv);
  container.scrollTop = container.scrollHeight;
  return id;
}

function removeLoadingMessage(id) {
  const elem = document.getElementById(id);
  if (elem) elem.remove();
}

function appendAssistantMessage(answerMarkdown, sources, modelUsed) {
  const container = document.getElementById("chatMessages");
  const msgDiv = document.createElement("div");
  msgDiv.className = "message assistant-message";

  const bubble = document.createElement("div");
  bubble.className = "message-bubble";

  // Parse markdown if marked is loaded
  if (window.marked) {
    bubble.innerHTML = marked.parse(answerMarkdown);
  } else {
    bubble.innerText = answerMarkdown;
  }

  // Render sources if available
  if (sources && sources.length > 0) {
    const sourcesBox = document.createElement("div");
    sourcesBox.className = "sources-box";

    const toggleBtn = document.createElement("button");
    toggleBtn.type = "button";
    toggleBtn.className = "sources-toggle";
    toggleBtn.innerHTML = `📚 <span>${sources.length} Context Sources Used</span> (Click to inspect)`;
    
    const sourcesList = document.createElement("div");
    sourcesList.className = "sources-list";
    sourcesList.style.display = "none";

    sources.forEach((src, idx) => {
      const srcItem = document.createElement("div");
      srcItem.className = "source-item";
      srcItem.innerHTML = `
        <div class="source-item-header">
          <span>#${idx + 1}: ${src.file_name} (Chunk #${src.chunk_index})</span>
          <span class="source-item-score">Cosine: ${src.score}</span>
        </div>
        <div class="source-item-text">${escapeHtml(src.text.slice(0, 260))}...</div>
      `;
      sourcesList.appendChild(srcItem);
    });

    toggleBtn.onclick = () => {
      const isHidden = sourcesList.style.display === "none";
      sourcesList.style.display = isHidden ? "flex" : "none";
      toggleBtn.querySelector("span").innerText = isHidden
        ? `${sources.length} Context Sources (Expanded)`
        : `${sources.length} Context Sources Used`;
    };

    sourcesBox.appendChild(toggleBtn);
    sourcesBox.appendChild(sourcesList);
    bubble.appendChild(sourcesBox);
  }

  msgDiv.appendChild(bubble);
  container.appendChild(msgDiv);
  container.scrollTop = container.scrollHeight;
}

function clearChat() {
  conversationHistory = [];
  const container = document.getElementById("chatMessages");
  container.innerHTML = `
    <div class="message system-message">
      <div class="message-bubble">
        <strong>🤖 Chat cleared!</strong>
        <p>Ask a new question to retrieve context and chat with the RAG agent.</p>
      </div>
    </div>
  `;
}

// --- 2. SIMILARITY SEARCH TAB LOGIC ---
async function handleSearchSubmit(event) {
  event.preventDefault();
  const input = document.getElementById("searchInput");
  const topK = parseInt(document.getElementById("searchTopK").value, 10) || 5;
  const submitBtn = document.getElementById("searchSubmitBtn");
  const query = input.value.trim();

  if (!query) return;

  submitBtn.disabled = true;
  submitBtn.innerHTML = '<span class="spinner"></span> Searching...';

  const emptyState = document.getElementById("searchEmptyState");
  const resultsList = document.getElementById("searchResultsList");

  try {
    const res = await fetch("/api/search", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query: query, top_k: topK })
    });

    if (!res.ok) throw new Error("Search request failed");
    const data = await res.json();

    if (!data.results || data.results.length === 0) {
      emptyState.style.display = "block";
      emptyState.innerHTML = `<div class="empty-icon">⚠️</div><p>No matching vectors found for "${escapeHtml(query)}". Try ingesting documents first.</p>`;
      resultsList.style.display = "none";
    } else {
      emptyState.style.display = "none";
      resultsList.style.display = "flex";
      resultsList.innerHTML = "";

      data.results.forEach((item, index) => {
        const card = document.createElement("div");
        card.className = "result-card";
        card.innerHTML = `
          <div class="result-card-header">
            <div class="result-file-tag">
              <span>📄</span>
              <span>${escapeHtml(item.file_name)} &bull; Chunk #${item.chunk_index}</span>
            </div>
            <div class="result-score-badge">Similarity: ${item.score}</div>
          </div>
          <div class="result-text">${escapeHtml(item.text)}</div>
        `;
        resultsList.appendChild(card);
      });
    }
  } catch (err) {
    emptyState.style.display = "block";
    emptyState.innerHTML = `<div class="empty-icon">❌</div><p>Error: ${err.message}</p>`;
    resultsList.style.display = "none";
  } finally {
    submitBtn.disabled = false;
    submitBtn.innerHTML = '<span>Search Vector DB</span>';
  }
}

// --- 3. INGESTION & UPLOAD LOGIC ---
function initDropzone() {
  const dropzone = document.getElementById("dropzone");
  if (!dropzone) return;

  ["dragenter", "dragover"].forEach(eventName => {
    dropzone.addEventListener(eventName, (e) => {
      e.preventDefault();
      dropzone.classList.add("dragover");
    }, false);
  });

  ["dragleave", "drop"].forEach(eventName => {
    dropzone.addEventListener(eventName, (e) => {
      e.preventDefault();
      dropzone.classList.remove("dragover");
    }, false);
  });

  dropzone.addEventListener("drop", (e) => {
    const dt = e.dataTransfer;
    const files = dt.files;
    if (files.length > 0) {
      handleFile(files[0]);
    }
  });
}

function handleFileSelected(event) {
  const file = event.target.files[0];
  if (file) handleFile(file);
}

function handleFile(file) {
  selectedFile = file;
  const fileInfo = document.getElementById("uploadFileInfo");
  const fileName = document.getElementById("uploadFileName");
  const statusMsg = document.getElementById("uploadStatus");

  fileName.innerText = `${file.name} (${(file.size / 1024).toFixed(1)} KB)`;
  fileInfo.style.display = "flex";
  statusMsg.style.display = "none";
}

async function submitFileUpload() {
  if (!selectedFile) return;

  const uploadBtn = document.getElementById("uploadBtn");
  const statusMsg = document.getElementById("uploadStatus");

  uploadBtn.disabled = true;
  uploadBtn.innerHTML = '<span class="spinner"></span> Ingesting...';
  statusMsg.style.display = "none";

  const formData = new FormData();
  formData.append("file", selectedFile);

  try {
    const res = await fetch("/api/upload", {
      method: "POST",
      body: formData
    });

    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Upload failed");

    statusMsg.className = "status-msg success";
    statusMsg.innerText = `✓ ${data.message}`;
    document.getElementById("uploadFileInfo").style.display = "none";
    selectedFile = null;

    // Refresh vector count
    fetchStatus();
  } catch (err) {
    statusMsg.className = "status-msg error";
    statusMsg.innerText = `✗ ${err.message}`;
  } finally {
    uploadBtn.disabled = false;
    uploadBtn.innerText = "Ingest Now";
  }
}

async function triggerBatchIngest() {
  const btn = document.getElementById("batchIngestBtn");
  const resultBox = document.getElementById("batchIngestResult");

  btn.disabled = true;
  btn.innerHTML = '<span class="spinner"></span> Processing & Embedding Documents...';
  resultBox.style.display = "none";

  try {
    const res = await fetch("/api/ingest", { method: "POST" });
    const data = await res.json();

    if (!res.ok) throw new Error(data.detail || "Ingestion failed");

    resultBox.style.display = "block";
    let summaryHtml = `<strong>✓ Ingestion Complete!</strong><br>${data.message}<br><br>`;
    
    if (data.details && data.details.processed_files) {
      summaryHtml += `<ul style="padding-left: 18px; margin-top: 6px;">`;
      data.details.processed_files.forEach(f => {
        summaryHtml += `<li><strong>${escapeHtml(f.file_name)}</strong>: ${f.chunks} chunks stored</li>`;
      });
      summaryHtml += `</ul>`;
    }

    resultBox.innerHTML = summaryHtml;
    fetchStatus();
  } catch (err) {
    resultBox.style.display = "block";
    resultBox.innerHTML = `<span style="color: #ef4444;">❌ Error: ${err.message}</span>`;
  } finally {
    btn.disabled = false;
    btn.innerHTML = '<span>⚡ Ingest All Documents in DATA/</span>';
  }
}

// Utility: Escape HTML
function escapeHtml(text) {
  if (!text) return "";
  const map = {
    '&': '&amp;',
    '<': '&lt;',
    '>': '&gt;',
    '"': '&quot;',
    "'": '&#039;'
  };
  return text.replace(/[&<>"']/g, m => map[m]);
}
