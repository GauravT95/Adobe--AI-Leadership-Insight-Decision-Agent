/**
 * AI Leadership Insight Agent — Frontend v2
 * Single-view with method dropdown, collapsible metrics, info panel
 */
(function () {
    'use strict';

    const API_BASE = '';

    // ═══ DOM refs ═══
    const chatForm     = document.getElementById('chat-form');
    const chatQuestion = document.getElementById('chat-question');
    const chatArea     = document.getElementById('chat-area');
    const sendBtn      = document.getElementById('send-btn');
    const newChatBtn   = document.getElementById('new-chat-btn');
    const sampleBtns   = document.querySelectorAll('.sample-q');
    const methodSelect = document.getElementById('method-select');
    const infoBtn      = document.getElementById('info-btn');
    const infoPanel    = document.getElementById('method-info-panel');
    const infoClose    = document.getElementById('info-close');

    const quickStartBtn  = document.getElementById('quick-start-btn');
    const quickStartPop  = document.getElementById('quick-start-popover');
    const popoverClose   = document.getElementById('popover-close');
    const statusDot      = document.getElementById('status-dot');

    let isLoading = false;

    // ═══ Quick Start popover ═══
    quickStartBtn?.addEventListener('click', () => {
        quickStartPop.classList.toggle('open');
    });

    popoverClose?.addEventListener('click', () => {
        quickStartPop.classList.remove('open');
    });

    // Close popover when clicking outside
    document.addEventListener('click', (e) => {
        if (quickStartPop && !quickStartPop.contains(e.target) && e.target !== quickStartBtn && !quickStartBtn.contains(e.target)) {
            quickStartPop.classList.remove('open');
        }
    });

    // ═══ Info panel toggle ═══
    infoBtn?.addEventListener('click', () => {
        infoPanel.style.display = infoPanel.style.display === 'none' ? 'block' : 'none';
    });

    infoClose?.addEventListener('click', () => {
        infoPanel.style.display = 'none';
    });

    // ═══ Health check (status dot only) ═══
    async function loadStatus() {
        try {
            const res = await fetch(API_BASE + '/health');
            const d = await res.json();
            if (statusDot) {
                statusDot.classList.toggle('active', d.status === 'healthy');
                statusDot.classList.toggle('offline', d.status !== 'healthy');
            }
        } catch {
            if (statusDot) {
                statusDot.classList.remove('active');
                statusDot.classList.add('offline');
            }
        }
    }

    // ═══ Method name mapper ═══
    const METHOD_LABELS = {
        hybrid_reranked: 'Hybrid + Reranker',
        agentic: 'Agentic (LangGraph)',
        hybrid: 'Hybrid RRF',
        naive_dense: 'Naive Dense (FAISS)',
        bm25: 'BM25 Sparse',
    };

    // ═══ Build chat message ═══
    function createMsg(role, html, meta) {
        const wrap = document.createElement('div');
        wrap.className = 'chat-message ' + role;

        const label = document.createElement('span');
        label.className = 'chat-label';
        label.textContent = role === 'user' ? 'You' : 'AI Agent';
        wrap.appendChild(label);

        const body = document.createElement('p');
        body.innerHTML = html;
        wrap.appendChild(body);

        // Collapsible metrics for AI responses
        if (meta && role === 'ai') {
            const methodLabel = METHOD_LABELS[meta.method] || meta.method;
            const confPct = meta.confidence !== undefined ? (meta.confidence * 100).toFixed(0) + '%' : '—';

            // Toggle bar (always visible)
            const toggle = document.createElement('div');
            toggle.className = 'metrics-toggle';
            toggle.innerHTML = '<span class="toggle-arrow">&#9654;</span>'
                + '<span class="metrics-summary">'
                + '<span class="pill">' + methodLabel + '</span>'
                + '<span class="pill">' + (meta.latency || 0) + 's</span>'
                + (meta.cache_hit ? '<span class="pill cache-hit">Cache Hit</span>' : '')
                + '</span>';
            wrap.appendChild(toggle);

            // Detail panel (hidden until clicked)
            const detail = document.createElement('div');
            detail.className = 'metrics-detail';
            detail.innerHTML = ''
                + row('Method', methodLabel)
                + row('Latency', (meta.latency || 0) + ' seconds')
                + row('Sources', (meta.sources && meta.sources.length) ? meta.sources.join(', ') : '—')
                + row('Cache Hit', meta.cache_hit ? 'Yes — matched: "' + esc(meta.matched_query || '') + '"' : 'No')
                + row('Scoring', 'LLM self-evaluation (quality_check node). Formal benchmarks use RAGAS framework — Faithfulness, Answer Relevance, Context Precision.');
            wrap.appendChild(detail);

            // Toggle click handler
            toggle.addEventListener('click', () => {
                toggle.classList.toggle('open');
                detail.classList.toggle('open');
            });
        }

        return wrap;
    }

    function row(label, value) {
        return '<div class="detail-row"><span class="detail-label">' + label + '</span><span class="detail-value">' + value + '</span></div>';
    }

    function showTyping() {
        const el = document.createElement('div');
        el.className = 'typing-indicator';
        el.id = 'typing-indicator';
        el.innerHTML = '<div class="typing-dot"></div><div class="typing-dot"></div><div class="typing-dot"></div>';
        chatArea.appendChild(el);
        chatArea.scrollTop = chatArea.scrollHeight;
    }

    function hideTyping() {
        const el = document.getElementById('typing-indicator');
        if (el) el.remove();
    }

    // ═══ Ask backend ═══
    async function askBackend(question, method) {
        const res = await fetch(API_BASE + '/ask', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ question, method }),
        });
        if (!res.ok) {
            const err = await res.json().catch(() => ({}));
            throw new Error(err.detail || 'Request failed');
        }
        return await res.json();
    }

    // ═══ Submit question ═══
    chatForm?.addEventListener('submit', async (e) => {
        e.preventDefault();
        const question = chatQuestion.value.trim();
        if (!question || isLoading) return;

        isLoading = true;
        sendBtn.disabled = true;
        chatQuestion.value = '';

        // Sample questions always remain visible

        chatArea.appendChild(createMsg('user', esc(question)));
        showTyping();

        try {
            const data = await askBackend(question, methodSelect.value);
            hideTyping();
            chatArea.appendChild(createMsg('ai', formatAnswer(data.answer), {
                method: data.method,
                latency: data.latency_seconds,
                confidence: data.confidence,
                sources: data.sources,
                cache_hit: data.cache_hit,
                matched_query: data.matched_query,
                question_type: data.question_type,
            }));
        } catch (err) {
            hideTyping();
            chatArea.appendChild(createMsg('ai', '<em>Error: ' + esc(err.message) + '</em>'));
        }

        chatArea.scrollTop = chatArea.scrollHeight;
        isLoading = false;
        sendBtn.disabled = false;
        chatQuestion.focus();
    });

    // ═══ Sample question clicks ═══
    sampleBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            chatQuestion.value = btn.textContent;
            chatForm.dispatchEvent(new Event('submit'));
        });
    });

    // ═══ New chat ═══
    newChatBtn?.addEventListener('click', () => {
        chatArea.innerHTML = '';
        chatQuestion.value = '';
    });

    // ═══ Utilities ═══
    function esc(str) {
        const d = document.createElement('div');
        d.textContent = str;
        return d.innerHTML;
    }

    function formatAnswer(text) {
        // Strip [Source: filename] citations — already shown in metrics
        var clean = text.replace(/\[Source:\s*[^\]]+\]/gi, '').replace(/\n\s*\n/g, '\n');
        return esc(clean)
            .replace(/## (.+)/g, '<strong class="answer-heading">$1</strong>')
            .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
            .replace(/- (.+)/g, '<span class="answer-bullet">$1</span>')
            .replace(/\n/g, '<br>');
    }

    // ═══ Init ═══
    loadStatus();

    // ═══ File upload ═══
    const fileInput = document.getElementById('file-upload');
    const uploadStatus = document.getElementById('upload-status');

    fileInput?.addEventListener('change', async () => {
        const file = fileInput.files[0];
        if (!file) return;

        uploadStatus.style.display = 'block';
        uploadStatus.className = 'upload-status uploading';
        uploadStatus.textContent = 'Uploading ' + file.name + '…';

        const formData = new FormData();
        formData.append('file', file);

        try {
            const res = await fetch(API_BASE + '/upload', {
                method: 'POST',
                body: formData,
            });
            const data = await res.json();
            if (!res.ok) throw new Error(data.detail || 'Upload failed');

            uploadStatus.className = 'upload-status success';
            uploadStatus.textContent = file.name + ' added — ' + data.chunks_added + ' chunks indexed. Total: ' + data.total_chunks;

            // Refresh status dot
            loadStatus();
        } catch (err) {
            uploadStatus.className = 'upload-status error';
            uploadStatus.textContent = 'Upload failed: ' + err.message;
        }

        // Clear input so same file can be re-uploaded
        fileInput.value = '';

        // Auto-hide after 8s
        setTimeout(() => { uploadStatus.style.display = 'none'; }, 8000);
    });

})();
