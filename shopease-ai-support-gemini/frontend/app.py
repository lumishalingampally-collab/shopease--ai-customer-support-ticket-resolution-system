"""ShopEase AI Support — polished Streamlit customer + agent console."""

import os
from pathlib import Path
from datetime import datetime
import requests
import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv
from voice_utils import transcribe_with_groq, translate_text, synthesize_gtts, talking_avatar_html, avatar_idle_html
load_dotenv(Path(__file__).resolve().parent.parent / '.env')

st.set_page_config(
    page_title="ShopEase AI Support",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_URL_DEFAULT = os.getenv("API_BASE_URL", "http://127.0.0.1:8001")

# ---------- Theme ----------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@600;700;800&display=swap');
:root { --ink:#182230; --muted:#667085; --brand:#6d5dfc; --brand2:#8b7cff; --soft:#f6f7fb; --line:#e8eaf0; --success:#13795b; --danger:#c0392b; }
html, body, [class*="css"] { font-family:'DM Sans', sans-serif; color:var(--ink); }
[data-testid="stAppViewContainer"] { background:linear-gradient(180deg,#fbfbfe 0%,#f6f7fb 100%); }
[data-testid="stSidebar"] { background:#111827; }
[data-testid="stSidebar"] * { color:#f3f4f6 !important; }
[data-testid="stSidebar"] .stTextInput input { background:#1f2937; border-color:#374151; }
.hero { padding:28px 30px; border-radius:24px; background:linear-gradient(135deg,#171b35 0%,#4038a6 55%,#7668ff 100%); color:white; box-shadow:0 16px 40px rgba(55,48,163,.18); margin-bottom:20px; }
.hero h1 { font-family:'Manrope',sans-serif; font-size:34px; margin:0 0 6px; letter-spacing:-1px; }
.hero p { margin:0; opacity:.86; font-size:15px; }
.badge { display:inline-block; padding:5px 10px; border-radius:999px; background:rgba(255,255,255,.13); font-size:12px; margin:0 5px 8px 0; }
.card { background:white; border:1px solid var(--line); border-radius:18px; padding:18px; box-shadow:0 5px 20px rgba(16,24,40,.04); }
.metric { background:white; border:1px solid var(--line); border-radius:16px; padding:16px 18px; }
.metric .label { color:var(--muted); font-size:12px; text-transform:uppercase; letter-spacing:.5px; }
.metric .value { font-family:'Manrope'; font-size:24px; font-weight:800; margin-top:4px; }
.source { background:#f8f8ff; border:1px solid #e6e3ff; border-radius:12px; padding:10px 12px; margin:7px 0; }
.ticket-open { border-left:4px solid #6d5dfc; }
.ticket-escalated { border-left:4px solid #e74c3c; }
.ticket-resolved { border-left:4px solid #20a070; }
.small { color:var(--muted); font-size:12px; }
.quick { background:#fff; border:1px solid var(--line); border-radius:14px; padding:12px; }
.stButton > button { border-radius:10px; font-weight:600; }
[data-testid="stChatMessage"] { border-radius:16px; transition:transform .2s ease, box-shadow .2s ease; }
[data-testid="stChatMessage"]:hover { transform:translateY(-2px); box-shadow:0 10px 25px rgba(16,24,40,.06); }
.voice-card { background:linear-gradient(135deg,#ffffff,#f7f5ff); border:1px solid #e5e1ff; border-radius:20px; padding:18px; margin:8px 0 18px; }
.voice-title { font-family:'Manrope'; font-size:18px; font-weight:800; margin-bottom:3px; }
.voice-sub { color:#667085; font-size:13px; margin-bottom:12px; }
.pulse-dot { display:inline-block; width:9px; height:9px; border-radius:50%; background:#6d5dfc; box-shadow:0 0 0 0 rgba(109,93,252,.45); animation:pulseDot 1.8s infinite; margin-right:7px; }
@keyframes pulseDot { 0%{box-shadow:0 0 0 0 rgba(109,93,252,.45)} 70%{box-shadow:0 0 0 10px rgba(109,93,252,0)} 100%{box-shadow:0 0 0 0 rgba(109,93,252,0)} }
</style>
""", unsafe_allow_html=True)


def api_get(path, **kwargs):
    return requests.get(f"{st.session_state.api_url}{path}", **kwargs)


def api_post(path, **kwargs):
    return requests.post(f"{st.session_state.api_url}{path}", **kwargs)


def api_patch(path, **kwargs):
    return requests.patch(f"{st.session_state.api_url}{path}", **kwargs)


if "api_url" not in st.session_state:
    st.session_state.api_url = API_URL_DEFAULT
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hi! I'm ShopEase AI Support. Tell me what happened and I'll look up the official ShopEase policy before answering."}]
if "active_ticket_id" not in st.session_state:
    st.session_state.active_ticket_id = None
if "active_ticket_number" not in st.session_state:
    st.session_state.active_ticket_number = None
if "last_chat" not in st.session_state:
    st.session_state.last_chat = None
if "answer_audio" not in st.session_state:
    st.session_state.answer_audio = None

# ---------- Sidebar ----------
with st.sidebar:
    st.markdown("## 🛍️ ShopEase")
    st.caption("AI Customer Support Console")
    st.markdown("---")
    st.session_state.api_url = st.text_input("Backend URL", st.session_state.api_url)
    try:
        health = api_get("/health", timeout=2)
        if health.ok:
            st.success("Backend online")
        else:
            st.warning(f"Backend HTTP {health.status_code}")
    except Exception:
        st.error("Backend offline")
    st.markdown("---")
    # Persistent-ticket resume control. Tickets and every message live in SQLite,
    # so browser/Streamlit refreshes do not delete the conversation.
    try:
        saved_resp = api_get("/api/v1/tickets?limit=20", timeout=5)
        saved_tickets = saved_resp.json() if saved_resp.ok else []
        ticket_options = ["Start a new conversation"] + [
            f"{t.get('ticket_number', '—')} · {t.get('category', 'OTHER')} · {t.get('status', 'OPEN')}"
            for t in saved_tickets
        ]
        selected_ticket = st.selectbox("Saved conversations", ticket_options, key="saved_ticket_selector")
        if selected_ticket != "Start a new conversation":
            selected_number = selected_ticket.split(" · ", 1)[0]
            if st.button("↩️ Resume selected ticket", use_container_width=True):
                detail = api_get(f"/api/v1/tickets/{selected_number}", timeout=8)
                if detail.ok:
                    ticket_data = detail.json()
                    st.session_state.active_ticket_id = ticket_data.get("id")
                    st.session_state.active_ticket_number = ticket_data.get("ticket_number")
                    st.session_state.messages = [
                        {"role": "assistant", "content": "Conversation resumed from saved ticket history."}
                    ]
                    for msg in ticket_data.get("messages", []):
                        role = "user" if msg.get("sender") == "USER" else "assistant"
                        st.session_state.messages.append({
                            "role": role,
                            "content": msg.get("content", "")
                        })
                    st.session_state.last_chat = {
                        "classification": {
                            "category": ticket_data.get("category", "OTHER"),
                            "priority": ticket_data.get("priority", "MEDIUM"),
                            "intent": ticket_data.get("intent", "general_question"),
                            "sentiment": ticket_data.get("sentiment", "NEUTRAL"),
                            "language": ticket_data.get("detected_language", "English"),
                        },
                        "needs_human_review": ticket_data.get("requires_human", False),
                        "status": ticket_data.get("status"),
                    }
                    st.rerun()
                else:
                    st.error("Could not load the saved ticket.")
    except Exception:
        # Backend health/status below gives the visible connection error.
        pass

    st.markdown("---")
    st.markdown("**Capabilities**")
    st.markdown("🧠 Issue analysis\n\n📚 Grounded knowledge retrieval\n\n🚨 Human escalation\n\n💬 Conversation history\n\n🌐 English · हिन्दी · తెలుగు")
    st.markdown("---")
    st.caption("ShopEase AI Support • Interview Demo")

# ---------- Header ----------
st.markdown("""
<div class="hero">
  <span class="badge">AI CUSTOMER SUPPORT</span><span class="badge">RAG GROUNDED</span><span class="badge">HUMAN-IN-THE-LOOP</span>
  <h1>Support that understands the customer.</h1>
  <p>Analyze the issue, retrieve official ShopEase policy, answer safely, and escalate when a specialist is needed.</p>
</div>
""", unsafe_allow_html=True)

# ---------- Navigation ----------
tab_chat, tab_dashboard, tab_kb = st.tabs(["💬 Customer Support", "🎫 Agent Dashboard", "📚 Knowledge Center"])

# ===================== CHAT =====================
with tab_chat:
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown('<div class="metric"><div class="label">Current ticket</div><div class="value">{}</div></div>'.format(st.session_state.active_ticket_number or "—"), unsafe_allow_html=True)
    with m2:
        last = st.session_state.last_chat or {}
        st.markdown('<div class="metric"><div class="label">Category</div><div class="value">{}</div></div>'.format(last.get("classification", {}).get("category", "—")), unsafe_allow_html=True)
    with m3:
        st.markdown('<div class="metric"><div class="label">Priority</div><div class="value">{}</div></div>'.format(last.get("classification", {}).get("priority", "—")), unsafe_allow_html=True)
    with m4:
        state = "Human review" if last.get("needs_human_review") else "AI assisted"
        st.markdown('<div class="metric"><div class="label">Resolution mode</div><div class="value">{}</div></div>'.format(state), unsafe_allow_html=True)

    left, right = st.columns([2.15, 1], gap="large")
    with left:
        st.markdown("### 💬 Conversation")
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        st.markdown('<div class="voice-card"><div class="voice-title">🎙️ Talk to ShopEase</div><div class="voice-sub"><span class="pulse-dot"></span>Speak naturally in English, हिन्दी or తెలుగు. Voice is transcribed, translated for retrieval, then answered in your language.</div></div>', unsafe_allow_html=True)
        voice_col1, voice_col2 = st.columns([2.2, 1])
        with voice_col1:
            voice_audio = st.audio_input("Record your support issue", key="customer_voice")
        with voice_col2:
            voice_lang_choice = st.selectbox("Spoken language", ["Auto-detect", "English", "Hindi", "Telugu"], key="voice_language")

        if voice_audio is not None and st.button("✨ Process voice message", type="primary", use_container_width=True):
            try:
                lang_hint = None if voice_lang_choice == "Auto-detect" else {"English":"en","Hindi":"hi","Telugu":"te"}[voice_lang_choice]
                with st.status("Processing your voice…", expanded=True) as status:
                    st.write("🎙️ Converting speech to text with Groq Whisper…")
                    transcript, detected = transcribe_with_groq(voice_audio, lang_hint)
                    source_lang = voice_lang_choice if voice_lang_choice != "Auto-detect" else detected
                    st.write(f"📝 **Transcription:** {transcript}")
                    st.write(f"🌐 Detected language: **{source_lang}**")
                    retrieval_query = transcript
                    if source_lang != "English":
                        st.write("🔄 Translating for knowledge-base retrieval…")
                        retrieval_query, translation_engine = translate_text(transcript, source_lang, "English")
                        st.write(f"Translation engine: **{translation_engine}**")
                    st.write("🔎 Running ShopEase classification + RAG…")
                    payload = {"query": retrieval_query, "ticket_id": st.session_state.active_ticket_id,
                               "customer_name": st.session_state.get("customer_name", "Demo Customer"),
                               "customer_email": st.session_state.get("customer_email", "customer@example.com")}
                    resp = api_post("/api/v1/chat", json=payload, timeout=30)
                    if not resp.ok:
                        raise RuntimeError(f"Support service returned HTTP {resp.status_code}: {resp.text[:500]}")
                    data = resp.json()
                    answer = data.get("response", "No response returned.")
                    if source_lang != "English":
                        st.write("🌐 Translating the grounded answer back to your language…")
                        answer, _ = translate_text(answer, "English", source_lang)
                    data["response"] = answer
                    data.setdefault("classification", {})["language"] = source_lang
                    st.session_state.last_chat = data
                    st.session_state.active_ticket_id = data.get("ticket_id")
                    st.session_state.active_ticket_number = data.get("ticket_number")
                    st.session_state.messages.append({"role": "user", "content": f"🎙️ {transcript}"})
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                    st.session_state.answer_audio = synthesize_gtts(answer, source_lang)
                    status.update(label="Voice request completed", state="complete")
                st.rerun()
            except Exception as exc:
                st.error(f"Voice processing failed: {exc}")
                st.caption("You can still type the request below. Check GROQ_API_KEY in .env if transcription is unavailable.")

        quick_cols = st.columns(3)
        quick_questions = [
            "My card was charged twice for the same order.",
            "How long does standard delivery take?",
            "What is the return window for electronics?",
        ]
        for i, q in enumerate(quick_questions):
            with quick_cols[i]:
                if st.button(q, key=f"quick_{i}", use_container_width=True):
                    st.session_state.pending_prompt = q
                    st.rerun()

        prompt = st.chat_input("Describe your problem in English, Hindi, or Telugu…")
        if not prompt and st.session_state.get("pending_prompt"):
            prompt = st.session_state.pop("pending_prompt")

        if prompt:
            st.session_state.answer_audio = None
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            with st.chat_message("assistant"):
                with st.spinner("Analyzing → retrieving policy → preparing a grounded response…"):
                    try:
                        payload = {
                            "query": prompt,
                            "ticket_id": st.session_state.active_ticket_id,
                            "customer_name": st.session_state.get("customer_name", "Demo Customer"),
                            "customer_email": st.session_state.get("customer_email", "customer@example.com"),
                        }
                        resp = api_post("/api/v1/chat", json=payload, timeout=30)
                        if not resp.ok:
                            st.error(f"Support service returned HTTP {resp.status_code}: {resp.text}")
                        else:
                            data = resp.json()
                            st.session_state.last_chat = data
                            st.session_state.active_ticket_id = data.get("ticket_id")
                            st.session_state.active_ticket_number = data.get("ticket_number")
                            st.markdown(data.get("response", "No response returned."))
                            st.session_state.messages.append({"role": "assistant", "content": data.get("response", "No response returned.")})
                            st.rerun()
                    except Exception as exc:
                        st.error(f"Unable to reach the FastAPI backend: {exc}")

    with right:
        st.markdown("### 🤖 AI Concierge")
        avatar_lang = (st.session_state.last_chat or {}).get("classification", {}).get("language", "English")
        components.html(
            talking_avatar_html(st.session_state.answer_audio, avatar_lang)
            if st.session_state.answer_audio else avatar_idle_html(avatar_lang),
            height=355,
            scrolling=False,
        )
        st.markdown("### 👤 Customer")
        st.session_state.customer_name = st.text_input("Name", st.session_state.get("customer_name", "Demo Customer"))
        st.session_state.customer_email = st.text_input("Email", st.session_state.get("customer_email", "customer@example.com"))
        st.markdown("### 🎯 AI Analysis")
        data = st.session_state.last_chat or {}
        clf = data.get("classification", {})
        if clf:
            st.markdown(f"**Category**  `{clf.get('category','—')}`")
            st.markdown(f"**Priority**  `{clf.get('priority','—')}`")
            st.markdown(f"**Intent**  `{clf.get('intent','—')}`")
            st.markdown(f"**Sentiment**  `{clf.get('sentiment','—')}`")
            st.markdown(f"**Language**  `{clf.get('language','—')}`")
            if data.get("needs_human_review"):
                st.error("🚨 Human specialist review required")
                if clf.get("escalation_reason"):
                    st.caption(clf["escalation_reason"])
            else:
                st.success("✅ Suitable for AI-assisted resolution")
        else:
            st.info("Send a message to see classification and escalation analysis.")
        if data:
            st.markdown("### 🧠 Generation trace")
            if data.get("llm_used"):
                st.success(f"Gemini generated the response · model: {data.get('generation_model') or 'configured model'}")
            else:
                st.info(f"Response mode: {data.get('generation_method', 'local_kb_fallback')}")
            st.caption(f"Retrieval query sent to Chroma: {data.get('retrieval_query') or '—'}")
            st.caption("The LLM receives the retrieved policy chunks shown below; it is instructed to answer only from those chunks.")
        if data.get("used_knowledge_base"):
            st.markdown("### 🛡️ Grounding status")
            if data.get("sufficient_information"):
                st.success("✅ Answer generated from verified ShopEase KB evidence")
            else:
                st.warning("⚠️ The KB does not contain enough evidence; no unsupported answer was generated.")

        if data.get("retrieved_evidence"):
            st.markdown("### 🔎 Retrieved KB evidence")
            st.caption("These are the exact knowledge-base chunks retrieved before AI response generation.")
            for i, ev in enumerate(data["retrieved_evidence"], 1):
                with st.expander(f"{i}. {ev.get('reference_id','KB')} · {ev.get('section','Policy')} · relevance {ev.get('relevance_score',0):.3f}"):
                    st.caption(f"Source: {ev.get('source','unknown')}")
                    st.markdown(ev.get("content", ""))

        if data.get("policy_references"):
            st.markdown("### 📚 Policy evidence")
            st.caption("These are internal KB references generated from the supplied Markdown sections; they are traceability IDs, not external policy numbers.")
            for ref in data["policy_references"]:
                score = ref.get("relevance_score", 0)
                st.markdown(
                    f'<div class="source"><b>🔖 {ref.get("reference_id", "KB")}</b><br>📄 {ref.get("source", "unknown")}<br>§ {ref.get("section", "Unknown section")}<br><small>Relevance: {score:.3f}</small></div>',
                    unsafe_allow_html=True,
                )
        elif data.get("sources"):
            st.markdown("### 📚 Verified sources")
            for source in data["sources"]:
                st.markdown(f'<div class="source">📄 {source}</div>', unsafe_allow_html=True)
        if st.button("＋ New conversation", use_container_width=True):
            st.session_state.messages = [{"role": "assistant", "content": "New conversation started. How can I help?"}]
            st.session_state.active_ticket_id = None
            st.session_state.active_ticket_number = None
            st.session_state.last_chat = None
            st.session_state.answer_audio = None
            st.rerun()

# ===================== DASHBOARD =====================
with tab_dashboard:
    st.markdown("### 🎫 Agent Resolution Center")
    st.caption("Monitor tickets, prioritize escalations, and close resolved conversations.")
    f1, f2, f3 = st.columns(3)
    with f1: status = st.selectbox("Status", ["ALL", "OPEN", "IN_PROGRESS", "RESOLVED", "ESCALATED"])
    with f2: priority = st.selectbox("Priority", ["ALL", "LOW", "MEDIUM", "HIGH", "URGENT"])
    with f3: category = st.selectbox("Category", ["ALL", "PAYMENT", "RETURN_REFUND", "SHIPPING", "ACCOUNT", "ORDER", "PROMOTION", "PRODUCT", "OTHER"])
    params = {k:v for k,v in {"status":status, "priority":priority, "category":category}.items() if v != "ALL"}
    try:
        resp = api_get("/api/v1/tickets", params=params, timeout=8)
        tickets = resp.json() if resp.ok else []
        total = len(tickets)
        escalated = sum(1 for t in tickets if t.get("status") == "ESCALATED")
        resolved = sum(1 for t in tickets if t.get("status") == "RESOLVED")
        open_count = total - resolved
        a,b,c,d = st.columns(4)
        for col,label,val in [(a,"Visible tickets",total),(b,"Open",open_count),(c,"Escalated",escalated),(d,"Resolved",resolved)]:
            with col: st.markdown(f'<div class="metric"><div class="label">{label}</div><div class="value">{val}</div></div>', unsafe_allow_html=True)
        st.markdown("---")
        if not tickets:
            st.info("No tickets match these filters.")
        for t in tickets:
            cls = "ticket-escalated" if t.get("status") == "ESCALATED" else ("ticket-resolved" if t.get("status") == "RESOLVED" else "ticket-open")
            with st.container(border=True):
                x1,x2,x3 = st.columns([2.4,1.2,1.2])
                with x1:
                    st.markdown(f"**🎫 {t.get('ticket_number','—')}**  ·  {t.get('category','—')}")
                    st.caption(f"{t.get('customer_name') or 'Anonymous'} · {t.get('created_at','')}")
                with x2: st.metric("Priority", t.get("priority", "—"))
                with x3: st.metric("Status", t.get("status", "—"))
                if t.get("requires_human") or t.get("escalation_reason"):
                    st.warning(f"🚨 {t.get('escalation_reason') or 'Human review required.'}")
                ac1, ac2, ac3 = st.columns([1,1,3])
                with ac1:
                    if t.get("status") != "RESOLVED" and st.button("Resolve", key=f"resolve_{t['id']}", use_container_width=True):
                        api_patch(f"/api/v1/tickets/{t['id']}", json={"status":"RESOLVED"})
                        st.rerun()
                with ac2:
                    if t.get("status") != "ESCALATED" and st.button("Escalate", key=f"escalate_{t['id']}", use_container_width=True):
                        api_patch(f"/api/v1/tickets/{t['id']}", json={"status":"ESCALATED", "requires_human":True})
                        st.rerun()
    except Exception as exc:
        st.error(f"Could not load ticket dashboard: {exc}")

# ===================== KNOWLEDGE =====================
with tab_kb:
    st.markdown("### 📚 Knowledge Center")
    st.caption("Inspect the authoritative ShopEase policies and verify semantic retrieval before an answer is generated.")
    try:
        stat = api_get("/api/v1/kb/status", timeout=5)
        if stat.ok:
            info = stat.json()
            a,b,c = st.columns(3)
            with a: st.metric("Indexed chunks", info.get("total_chunks", 0))
            with b: st.metric("Retrieval threshold", info.get("threshold", "—"))
            with c: st.metric("Embedding model", info.get("embedding_model", "—"))
            st.success("Knowledge base is indexed" if info.get("is_indexed") else "Knowledge base needs indexing")
        if st.button("🔄 Reindex official policies", use_container_width=True):
            with st.spinner("Rebuilding semantic index…"):
                r = api_post("/api/v1/kb/reindex", timeout=30)
                if r.ok: st.success(r.json().get("message", "Reindexed successfully.")); st.rerun()
                else: st.error(r.text)
    except Exception as exc:
        st.warning(f"Knowledge service unavailable: {exc}")

    st.markdown("---")
    qcol, kcol = st.columns([4,1])
    with qcol: query = st.text_input("Semantic search", "My card was charged twice")
    with kcol: top_k = st.slider("Top K", 1, 5, 3)
    if st.button("🔎 Test retrieval", type="primary"):
        try:
            r = api_post("/api/v1/kb/search", json={"query":query,"top_k":top_k}, timeout=15)
            if r.ok:
                data=r.json()
                if data.get("has_sufficient_context"): st.success(f"{data['results_count']} relevant policy chunks found.")
                else: st.warning("No sufficiently relevant policy context found — response should not guess.")
                for i,res in enumerate(data.get("results",[]),1):
                    with st.expander(f"{i}. {res.get('topic','Policy')} · similarity {res.get('similarity_score',0):.1%}"):
                        st.caption(f"Source: {res.get('source')} · Category: {res.get('category')}")
                        st.markdown(res.get("content", ""))
            else: st.error(r.text)
        except Exception as exc: st.error(f"Retrieval failed: {exc}")

    st.markdown("---")
    st.markdown("### 📄 Official policy documents")
    kb_dir = Path(__file__).resolve().parent.parent / "knowledge_base"
    files = sorted(kb_dir.glob("*.md")) if kb_dir.exists() else []
    if files:
        names=[p.name for p in files]
        selected=st.selectbox("Choose a document", names)
        content=(kb_dir/selected).read_text(encoding="utf-8")
        st.markdown(content)
    else:
        st.warning("No knowledge-base documents found.")
