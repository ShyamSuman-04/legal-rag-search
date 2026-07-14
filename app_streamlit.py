"""
US Tax & Legal RAG — Production UI (Hugging Face Spaces build)
Harvey AI / Perplexity inspired Streamlit front end.

Run with:  streamlit run app.py

Deployment note
----------------
This is the "hf-deployment" branch variant of ui/app.py. On the main
branch, this UI talks to a separate FastAPI backend (api/main.py) over
HTTP. Hugging Face Spaces only runs a single process for a Streamlit
Space, so here the app imports RAGPipeline directly and calls it
in-process instead - no HTTP hop, no FastAPI, one fewer moving part
to keep alive on a free-tier Space.

    main branch:                Streamlit -> HTTP -> FastAPI -> RAGPipeline
    hf-deployment (this file):  Streamlit -> RAGPipeline directly

Nothing under rag/, retrieval/, ingestion/, indexing/, embeddings/, or
config.py changes for this branch - only this file. api/main.py and
api/schemas.py are simply unused here (safe to leave in the repo).
"""

import os
import time
import streamlit as st
import streamlit.components.v1 as components

from rag.rag_pipeline import RAGPipeline

# ==========================================================
# 1. CONFIG
# ==========================================================

APP_TITLE = "US Tax & Legal RAG"
APP_TAGLINE = "Hybrid Search • BM25 • Vector Search • CrossEncoder • Groq"
ASSISTANT_NAME = "Legal Research Assistant"

# Shown in the sidebar only. There's no live /stats endpoint anymore
# (see fetch_corpus_stats below), so these are static - override via
# a Space "Variable" (Settings -> Variables, not Secrets, since these
# aren't sensitive) if your real corpus size differs.
DOCUMENT_COUNT = int(os.environ.get("DOCUMENT_COUNT", "100"))
CHUNK_COUNT = int(os.environ.get("CHUNK_COUNT", "5428"))

SUGGESTED_QUESTIONS = [
    "What is IRC Section 162?",
    "What expenses qualify as ordinary and necessary business deductions?",
    "How does the statute of limitations apply to tax audits?",
    "What is the difference between a tax credit and a tax deduction?",
]

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==========================================================
# 2. RAG PIPELINE
# ==========================================================
# Loaded once per Space container and cached for the life of the
# process - this is what api/main.py did at module import time with
# `pipeline = RAGPipeline(debug=False)`; st.cache_resource is the
# Streamlit-native way to do the same "build once, reuse forever"
# thing for an object that isn't safe/cheap to hash or pickle (model
# weights, DB connections, an Elasticsearch/Neo4j client, etc.).


@st.cache_resource(show_spinner=False)
def load_pipeline() -> RAGPipeline:
    # NOTE: this assumes RAGPipeline.__init__ accepts a `debug` kwarg,
    # matching api/main.py's own `RAGPipeline(debug=False)` call. If
    # your RAGPipeline.__init__ signature doesn't take `debug`, just
    # call RAGPipeline() here instead.
    return RAGPipeline(debug=False)


try:
    with st.spinner(
        "🔧 Loading models and connecting to the knowledge base "
        "— this can take a minute on first load..."
    ):
        pipeline = load_pipeline()
    is_online = True
except Exception as exc:
    # st.cache_resource does not cache a failed call, so fixing the
    # underlying secret/connection issue and letting Streamlit rerun
    # (or the user refreshing) will retry cleanly - no need to clear
    # any cache by hand.
    st.error(
        "⚠️ Failed to initialize the RAG pipeline.\n\n"
        "Check your Space secrets (`GROQ_API_KEY`, Elasticsearch, "
        "Neo4j credentials, etc. - whatever `config.py` expects) and "
        "the Space logs for the full traceback.\n\n"
        f"**Error:** {exc}"
    )
    st.stop()

# ==========================================================
# 3. SESSION STATE
# ==========================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "last_question" not in st.session_state:
    st.session_state.last_question = None

if "pending_question" not in st.session_state:
    st.session_state.pending_question = None

# ==========================================================
# 4. THEME / CSS
# ==========================================================
# Design language: a legal "case file" — deep ink navy, aged brass /
# gold-leaf accents, warm ledger paper, faint ruled-paper texture.
# Display type is Fraunces (characterful serif for the letterhead),
# body/reading type is Source Serif 4, and stamps / metadata / citation
# numbers use IBM Plex Mono, the way a case file uses a typewriter face
# for docket numbers and exhibit tags.

st.markdown(
    """
<style>

@import url('https://fonts.googleapis.com/css2?family=Fraunces:ital,opsz,wght@0,9..144,400;0,9..144,500;0,9..144,600;0,9..144,700;1,9..144,500&family=Source+Serif+4:ital,wght@0,400;0,500;0,600;1,400&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

:root {
    --ink: #1B2A41;
    --ink-soft: #4B586C;
    --paper: #FAF6EC;
    --paper-dim: #F0E8D6;
    --panel-border: rgba(27, 42, 65, 0.14);
    --brass: #9C7A3C;
    --brass-bright: #B68B3E;
    --brass-wash: rgba(156, 122, 60, 0.12);
    --forest: #35493C;
    --burgundy: #7A3030;
    --shadow: rgba(27, 42, 65, 0.10);
}

@media (prefers-color-scheme: dark) {
    :root {
        --ink: #EDE6D6;
        --ink-soft: #B9B2A0;
        --paper: #10161F;
        --paper-dim: #19212D;
        --panel-border: rgba(201, 162, 39, 0.20);
        --brass: #C9A227;
        --brass-bright: #E0BC4E;
        --brass-wash: rgba(201, 162, 39, 0.12);
        --forest: #5C9270;
        --burgundy: #C97A7A;
        --shadow: rgba(0, 0, 0, 0.45);
    }
}

html, body, [class*="css"] {
    font-family: "Source Serif 4", "Iowan Old Style", Georgia, serif;
    color: var(--ink);
}

h1, h2, h3, .hero h1 {
    font-family: "Fraunces", "Georgia", serif;
}

#MainMenu, footer, header {visibility: hidden;}

[data-testid="stAppViewContainer"] {
    background-color: var(--paper);
}

[data-testid="stAppViewContainer"] > .main {
    background-image: repeating-linear-gradient(
        to bottom,
        transparent 0,
        transparent 27px,
        var(--panel-border) 27px,
        var(--panel-border) 28px
    );
    background-attachment: local;
}

[data-testid="stSidebar"] {
    background-color: var(--paper-dim);
    border-right: 1px solid var(--panel-border);
}

[data-testid="stSidebar"] * {
    color: var(--ink);
}

/* ---------- Hero / letterhead ---------- */

.hero {
    position: relative;
    padding: 1.5rem 1.9rem 1.2rem 1.9rem;
    border-radius: 4px;
    background: var(--paper-dim);
    border: 1px solid var(--panel-border);
    border-top: 3px solid var(--brass);
    box-shadow: 0 6px 18px var(--shadow);
    margin-bottom: 1.1rem;
}

.hero-top {
    display: flex;
    align-items: center;
    gap: 0.85rem;
}

.brand-mark {
    font-size: 1.9rem;
    line-height: 1;
    flex-shrink: 0;
}

.hero h1 {
    margin: 0;
    font-size: 1.85rem;
    font-weight: 600;
    letter-spacing: 0.2px;
    color: var(--ink);
}

.hero-tagline {
    margin: 0.15rem 0 0 0;
    font-style: italic;
    font-size: 0.98rem;
    color: var(--ink-soft);
}

.hero-stack {
    display: flex;
    flex-wrap: wrap;
    gap: 0.4rem;
    margin-top: 0.95rem;
    padding-top: 0.8rem;
    border-top: 1px dashed var(--panel-border);
}

.stack-chip {
    font-family: "IBM Plex Mono", monospace;
    font-size: 0.68rem;
    letter-spacing: 0.4px;
    text-transform: uppercase;
    padding: 0.22rem 0.55rem;
    border-radius: 999px;
    background: var(--brass-wash);
    border: 1px solid var(--panel-border);
    color: var(--brass);
}

.header-status {
    position: absolute;
    top: 1.1rem;
    right: 1.6rem;
    display: inline-flex;
    align-items: center;
    padding: 0.32rem 0.75rem;
    border-radius: 999px;
    background: var(--paper);
    border: 1px solid var(--panel-border);
    font-family: "IBM Plex Mono", monospace;
    font-size: 0.72rem;
    letter-spacing: 0.3px;
    text-transform: uppercase;
    box-shadow: 0 2px 6px var(--shadow);
}

.status-dot {
    height: 8px;
    width: 8px;
    border-radius: 50%;
    display: inline-block;
    margin-right: 6px;
}
.status-online { background-color: var(--forest); box-shadow: 0 0 0 3px color-mix(in srgb, var(--forest) 25%, transparent); }
.status-offline { background-color: var(--burgundy); box-shadow: 0 0 0 3px color-mix(in srgb, var(--burgundy) 25%, transparent); }

/* ---------- General cards / glass ---------- */

.glass-card {
    background: var(--paper-dim);
    border: 1px solid var(--panel-border);
    border-radius: 6px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.8rem;
}

.badge {
    display: inline-block;
    padding: 0.16rem 0.6rem;
    border-radius: 999px;
    background: var(--brass-wash);
    border: 1px solid var(--panel-border);
    font-size: 0.72rem;
    margin-right: 0.35rem;
    color: var(--ink-soft);
}

.badge-mono {
    font-family: "IBM Plex Mono", monospace;
    letter-spacing: 0.2px;
}

/* ---------- Citations / references — the case-file footnote ---------- */

.ref-card {
    display: flex;
    gap: 0.7rem;
    align-items: flex-start;
    border: 1px solid var(--panel-border);
    border-left: 3px solid var(--brass);
    border-radius: 4px;
    padding: 0.65rem 0.85rem;
    margin-bottom: 0.5rem;
    background: var(--paper);
}

.ref-num {
    font-family: "IBM Plex Mono", monospace;
    font-size: 0.72rem;
    font-weight: 600;
    color: var(--brass);
    padding-top: 0.1rem;
    flex-shrink: 0;
}

.ref-body b {
    font-family: "Fraunces", serif;
    font-weight: 500;
}

.ref-meta {
    margin-top: 0.3rem;
}

/* ---------- Chat bubbles (Streamlit native containers) ---------- */

[data-testid="stChatMessage"] {
    background: var(--paper-dim);
    border: 1px solid var(--panel-border);
    border-radius: 10px;
    box-shadow: 0 2px 8px var(--shadow);
    padding: 0.3rem 0.4rem;
}

/* ---------- Inputs, buttons, misc widgets ---------- */

.stChatInputContainer, .stButton>button, .stDownloadButton>button {
    border-radius: 8px !important;
}

.stButton>button, .stDownloadButton>button {
    font-family: "Source Serif 4", serif;
    border: 1px solid var(--panel-border) !important;
    background: var(--paper) !important;
    color: var(--ink) !important;
}

.stButton>button:hover, .stDownloadButton>button:hover {
    border-color: var(--brass) !important;
    color: var(--brass) !important;
}

[data-testid="stChatInput"] {
    border-radius: 10px;
    border: 1px solid var(--panel-border) !important;
}

[data-testid="stMetricValue"] {
    font-family: "IBM Plex Mono", monospace;
    color: var(--ink);
}

[data-testid="stMetricLabel"] {
    font-family: "IBM Plex Mono", monospace;
    text-transform: uppercase;
    font-size: 0.68rem;
    letter-spacing: 0.4px;
    color: var(--ink-soft);
}

[data-testid="stExpander"] {
    border: 1px solid var(--panel-border) !important;
    border-radius: 6px !important;
    background: var(--paper-dim);
}

hr {
    border-color: var(--panel-border) !important;
}

/* Sidebar section labels styled like docket headers */
.sidebar-label {
    font-family: "IBM Plex Mono", monospace;
    font-size: 0.7rem;
    letter-spacing: 0.6px;
    text-transform: uppercase;
    color: var(--brass);
    border-bottom: 1px solid var(--panel-border);
    padding-bottom: 0.3rem;
    margin-bottom: 0.5rem;
}

.spec-sheet {
    font-size: 0.88rem;
    line-height: 1.85;
}
.spec-sheet .spec-key {
    color: var(--ink-soft);
    font-family: "IBM Plex Mono", monospace;
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.3px;
    margin-right: 0.4rem;
}

</style>
""",
    unsafe_allow_html=True,
)

# ==========================================================
# 5. HELPERS
# ==========================================================


def fetch_corpus_stats():
    """Static corpus stats for the sidebar.

    There's no /stats endpoint to call anymore - this mirrors the
    same hardcoded values api/main.py's /stats route returned
    (document_count=100, chunk_count=5428), just sourced from the
    DOCUMENT_COUNT/CHUNK_COUNT constants above instead of an HTTP
    round-trip. Swap this for a real query later (e.g. an
    Elasticsearch count via the pipeline's own client) if you want it
    live instead of static.
    """
    return {"document_count": DOCUMENT_COUNT, "chunk_count": CHUNK_COUNT}


def call_ask_api(question: str):
    """Runs the RAG pipeline in-process and returns an HTTP-shaped result dict.

    Kept the name call_ask_api (rather than renaming to something like
    run_pipeline) so every other line in this file - all written
    against this exact result shape - keeps working unchanged.

    Wrapped in try/except because there's no FastAPI/HTTP layer
    catching connection errors or 500s for us anymore: a Groq
    timeout, an empty-question ValueError, an Elasticsearch hiccup,
    or a paused Neo4j AuraDB instance would otherwise surface as a
    raw Streamlit traceback instead of the friendly inline error
    bubble (see section 11 below) the rest of this UI already expects
    via result["error"].
    """
    try:
        response = pipeline.answer_question(question)
        return {
            "answer": response["answer"],
            "latency": response["latency_seconds"],
            "model": response["model"],
            "references": response.get("references", []),
            "error": None,
        }
    except Exception as exc:
        return {
            "answer": None,
            "latency": None,
            "model": None,
            "references": [],
            "error": f"Pipeline error: {exc}",
        }


def copy_to_clipboard_button(text: str, key: str):
    """Renders a 'Copy Answer' button using an embedded HTML component.
    Lawyers tend to copy answers into their own documents far more often
    than they download files, so this sits alongside the download buttons."""
    safe_text = text.replace("`", "\\`").replace("</", "<\\/")
    components.html(
        f"""
        <button id="copy-btn-{key}"
            style="
                background: rgba(156,122,60,0.12);
                border: 1px solid rgba(156,122,60,0.35);
                color: #9C7A3C;
                border-radius: 8px;
                padding: 7px 14px;
                font-size: 13px;
                width: 100%;
                cursor: pointer;
                font-family: Georgia, 'Source Serif 4', serif;
                transition: background 0.15s ease;
            "
            onmouseover="this.style.background='rgba(156,122,60,0.20)'"
            onmouseout="this.style.background='rgba(156,122,60,0.12)'"
            onclick="
                navigator.clipboard.writeText(`{safe_text}`);
                const b = document.getElementById('copy-btn-{key}');
                b.innerText = 'Copied ✓';
                setTimeout(() => {{ b.innerText = '📋 Copy Answer'; }}, 1500);
            "
        >📋 Copy Answer</button>
        """,
        height=42,
    )


def render_references(references):
    if not references:
        st.warning("No references were returned for this answer.")
        return
    with st.expander(f"📄 References ({len(references)})", expanded=False):
        for i, ref in enumerate(references, start=1):
            doc_name = ref.get("document_name", "Unknown document")
            doc_type = ref.get("document_type", "N/A")
            page_start = ref.get("page_start", "?")
            page_end = ref.get("page_end", "?")
            st.markdown(
                f"""
<div class="ref-card">
<span class="ref-num">{i:02d}</span>
<div class="ref-body">
<b>{doc_name}</b>
<div class="ref-meta">
<span class="badge">{doc_type}</span>
<span class="badge badge-mono">pp. {page_start}–{page_end}</span>
</div>
</div>
</div>
""",
                unsafe_allow_html=True,
            )


def render_assistant_message(message, index):
    with st.chat_message("assistant", avatar="🖋️"):
        st.caption(f"🖋️ {ASSISTANT_NAME}")
        st.markdown(message["content"])

        if message.get("latency") is not None or message.get("model"):
            m1, m2 = st.columns(2)
            with m1:
                if message.get("latency") is not None:
                    st.metric("⏱ Response Time", f"{message['latency']:.2f} sec")
            with m2:
                if message.get("model"):
                    st.metric("🤖 Model", message["model"])

        render_references(message.get("references", []))

        # Actions: copying tends to be used far more than downloading a file,
        # so it gets equal billing right next to the download buttons.
        act1, act2, act3 = st.columns(3)
        with act1:
            copy_to_clipboard_button(message["content"], key=f"copy-{index}")
        with act2:
            st.download_button(
                label="📥 Download (.md)",
                data=message["content"],
                file_name=f"legal_answer_{index}.md",
                mime="text/markdown",
                key=f"dl-md-{index}",
                use_container_width=True,
            )
        with act3:
            st.download_button(
                label="📥 Download (.txt)",
                data=message["content"],
                file_name=f"legal_answer_{index}.txt",
                mime="text/plain",
                key=f"dl-txt-{index}",
                use_container_width=True,
            )

        st.divider()


def render_user_message(message):
    with st.chat_message("user", avatar="👤"):
        st.markdown(message["content"])


# ==========================================================
# 6. SIDEBAR
# ==========================================================

with st.sidebar:
    st.markdown("### ⚖️ US Tax & Legal RAG")
    st.caption(APP_TAGLINE)
    st.caption("Running in-process on Hugging Face Spaces")
    st.divider()

    # Backend online/offline status now lives top-right in the header,
    # where it's far more visible than tucked away in the sidebar.
    stats = fetch_corpus_stats()
    if stats:
        st.markdown('<div class="sidebar-label">Corpus</div>', unsafe_allow_html=True)
        s1, s2 = st.columns(2)
        with s1:
            if "document_count" in stats:
                st.metric("Documents", stats["document_count"])
        with s2:
            if "chunk_count" in stats:
                st.metric("Chunks", stats["chunk_count"])

    st.divider()
    st.markdown('<div class="sidebar-label">Architecture</div>', unsafe_allow_html=True)
    st.markdown(
        """
<div class="spec-sheet">
<div><span class="spec-key">Retrieval</span>Hybrid (BM25 + Vector)</div>
<div><span class="spec-key">Reranking</span>CrossEncoder</div>
<div><span class="spec-key">Generation</span>Groq</div>
<div><span class="spec-key">Frontend</span>Streamlit (single-process)</div>
</div>
""",
        unsafe_allow_html=True,
    )

    st.divider()
    st.markdown('<div class="sidebar-label">Try asking</div>', unsafe_allow_html=True)
    for q in SUGGESTED_QUESTIONS:
        if st.button(q, key=f"suggest-{q}", use_container_width=True):
            st.session_state.pending_question = q

    st.divider()
    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.last_question = None
        st.rerun()

# ==========================================================
# 7. HEADER
# ==========================================================

status_class = "status-online" if is_online else "status-offline"
status_text = "System Ready" if is_online else "System Unavailable"

tagline_parts = [p.strip() for p in APP_TAGLINE.split("•") if p.strip()]
stack_chips_html = "".join(f'<span class="stack-chip">{part}</span>' for part in tagline_parts)

st.markdown(
    f"""
<div class="hero">
<span class="header-status">
    <span class="status-dot {status_class}"></span>{status_text}
</span>
<div class="hero-top">
<span class="brand-mark">⚖️</span>
<div>
<h1>{APP_TITLE}</h1>
<p class="hero-tagline">Grounded legal research, fully cited · Running in-process on Hugging Face Spaces</p>
</div>
</div>
<div class="hero-stack">{stack_chips_html}</div>
</div>
""",
    unsafe_allow_html=True,
)

# ==========================================================
# 8. EMPTY STATE
# ==========================================================

if len(st.session_state.messages) == 0:
    st.info(
        """
👋 **Welcome!** Ask any question regarding:

- Acts
- Court Judgments
- Tax Documents
- Legal Commentaries

Use the suggestions in the sidebar, or type your own question below.
"""
    )

# ==========================================================
# 9. CONVERSATION HISTORY
# ==========================================================

for idx, message in enumerate(st.session_state.messages):
    if message["role"] == "user":
        render_user_message(message)
    else:
        render_assistant_message(message, idx)

# ==========================================================
# 10. CHAT INPUT
# ==========================================================

question = st.chat_input("Ask a question about the legal documents...")

if st.session_state.pending_question:
    question = st.session_state.pending_question
    st.session_state.pending_question = None

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    st.rerun()

# ==========================================================
# 11. GENERATE AI RESPONSE (only for the newest, un-answered question)
# ==========================================================

if st.session_state.messages:
    last_message = st.session_state.messages[-1]

    if (
        last_message["role"] == "user"
        and last_message["content"] != st.session_state.last_question
    ):
        st.session_state.last_question = last_message["content"]

        start = time.time()
        with st.status("Searching legal documents...", expanded=True) as status:
            st.write("🔎 Running hybrid search (BM25 + vector)...")
            st.write("📚 Reranking with CrossEncoder...")
            st.write("🤖 Generating grounded answer...")
            result = call_ask_api(last_message["content"])
            elapsed = time.time() - start

            if result["error"]:
                status.update(label="Search failed", state="error", expanded=False)
            else:
                status.update(label="Answer ready", state="complete", expanded=False)

        if result["error"]:
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": f"⚠️ {result['error']}",
                    "latency": elapsed,
                    "model": None,
                    "references": [],
                }
            )
        else:
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": result["answer"],
                    "latency": result["latency"] if result["latency"] is not None else elapsed,
                    "model": result["model"],
                    "references": result["references"],
                }
            )

        st.rerun()

# ==========================================================
# 12. FOOTER
# ==========================================================

st.divider()
st.caption(
    f"""
⚖️ {APP_TITLE} · {APP_TAGLINE}

Built with Streamlit • Hybrid RAG • Groq
"""
)