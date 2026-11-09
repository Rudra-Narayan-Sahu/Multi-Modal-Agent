import streamlit as st
import time
import uuid
from app.agents.graph import run_agent_workflow
from app.router.engine import default_router
from app.router.models import RouterRequest
from app.memory.conversation import conversation_manager
from app.config import settings

st.set_page_config(
    page_title="Multi-Agent RAG Assistant (LangGraph)",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .badge-agent {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 14px;
        font-size: 0.8rem;
        font-weight: 600;
        background-color: #ede9fe;
        color: #5b21b6;
        margin-right: 6px;
        margin-bottom: 4px;
    }
    .badge-category {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 14px;
        font-size: 0.8rem;
        font-weight: 600;
        background-color: #e0f2fe;
        color: #0369a1;
        margin-right: 6px;
        margin-bottom: 4px;
    }
    .badge-status {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 14px;
        font-size: 0.8rem;
        font-weight: 600;
        background-color: #ecfdf5;
        color: #047857;
        margin-bottom: 4px;
    }
</style>
""", unsafe_allow_html=True)


# Initialize Session State
if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = f"session_{uuid.uuid4().hex[:8]}"

if "messages" not in st.session_state:
    st.session_state.messages = []


# Sidebar
with st.sidebar:
    st.title("🤖 Multi-Agent Control")
    st.caption("Orchestrated with LangGraph & Qdrant")

    st.markdown("---")
    st.subheader("Session Details")
    st.code(st.session_state.conversation_id, language="text")

    if st.button("🗑️ Clear Conversation History", use_container_width=True):
        conversation_manager.clear(st.session_state.conversation_id)
        st.session_state.messages = []
        st.session_state.conversation_id = f"session_{uuid.uuid4().hex[:8]}"
        st.rerun()

    st.markdown("---")
    st.subheader("Orchestration Mode")
    engine_mode = st.radio(
        "Select Execution Engine",
        ["LangGraph Multi-Agent (6 Agents)", "Direct Query Router (Single-Turn)"],
        index=0
    )

    st.markdown("---")
    st.markdown("### 👥 Active LangGraph Agents")
    st.markdown("""
    1. **Supervisor Agent** (Planning & Routing)
    2. **RAG Agent** (Qdrant Cloud & FlashRank)
    3. **Web Search Agent** (Tavily & DuckDuckGo)
    4. **Research Agent** (Cross-Synthesis & Gaps)
    5. **Tool Agent** (Calculator, Python, DateTime)
    6. **Response Agent** (Grounded Answer & Citations)
    """)

    st.markdown("---")
    show_agent_trace = st.toggle("Show Agent Execution Logs", value=True)


# Main Interface Header
st.title("🤖 Enterprise Multi-Agent Research Assistant")
st.caption("Powered by LangGraph multi-agent state machines, Tavily web search, and Qdrant vector retrieval.")

# Render Conversation
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        
        # Badges & metadata
        if msg["role"] == "assistant" and "meta" in msg:
            meta = msg["meta"]
            category = meta.get("category", "")
            agents = meta.get("agents", [])
            status = meta.get("status", "completed")

            badges_html = f'<div style="margin-top: 8px; margin-bottom: 6px;">'
            if category:
                badges_html += f'<span class="badge-category">Category: {category}</span>'
            for a in agents:
                badges_html += f'<span class="badge-agent">🤖 {a}</span>'
            badges_html += f'<span class="badge-status">Status: {status}</span></div>'
            st.markdown(badges_html, unsafe_allow_html=True)

            # Execution trace expander
            if show_agent_trace and meta.get("agent_messages"):
                with st.expander("🔍 Multi-Agent Execution Trail", expanded=False):
                    for am in meta.get("agent_messages", []):
                        st.json(am)

            # Citations expander
            if meta.get("citations"):
                with st.expander("📚 Source Citations & Provenance", expanded=False):
                    for c in meta.get("citations", []):
                        if c.get("type") == "document":
                            st.markdown(f"- 📄 **{c.get('file_name')}** (Chunk #{c.get('chunk_index')}, Score: {c.get('score')})")
                        elif c.get("type") == "web":
                            st.markdown(f"- 🌐 [{c.get('title')}]({c.get('url')})")


# User Input
if prompt := st.chat_input("Ask a research question, compare topics, or request a calculation..."):
    # Append user prompt
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Execute Assistant
    with st.chat_message("assistant"):
        start_time = time.time()
        
        if "LangGraph" in engine_mode:
            with st.spinner("Multi-Agent Graph is coordinating (Supervisor -> RAG / Web / Tool -> Research -> Response)..."):
                final_state = run_agent_workflow(
                    query=prompt,
                    conversation_id=st.session_state.conversation_id,
                    user_id="default_user"
                )
            
            elapsed = time.time() - start_time
            answer = final_state.get("final_response", "No response generated.")
            category = final_state.get("query_category", "UNKNOWN")
            agents = final_state.get("selected_agents", [])
            citations = final_state.get("source_references", [])
            messages = final_state.get("agent_messages", [])
            status = final_state.get("execution_status", "completed")

            st.markdown(answer)

            # Metadata Badges
            badges_html = f'<div style="margin-top: 8px; margin-bottom: 6px;">'
            badges_html += f'<span class="badge-category">Category: {category}</span>'
            for a in agents:
                badges_html += f'<span class="badge-agent">🤖 {a}</span>'
            badges_html += f'<span class="badge-status">Completed in {elapsed:.2f}s</span></div>'
            st.markdown(badges_html, unsafe_allow_html=True)

            if show_agent_trace and messages:
                with st.expander("🔍 Multi-Agent Execution Trail", expanded=False):
                    for am in messages:
                        st.json(am)

            # Store in session state
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "meta": {
                    "category": category,
                    "agents": agents,
                    "citations": citations,
                    "agent_messages": messages,
                    "status": status,
                    "elapsed": elapsed
                }
            })

        else:
            with st.spinner("Processing through single-turn query router..."):
                req = RouterRequest(
                    query=prompt,
                    conversation_id=st.session_state.conversation_id,
                    execute_handler=True
                )
                router_res = default_router.process(req)

            elapsed = time.time() - start_time
            answer = router_res.handler_response or "No response generated."
            st.markdown(answer)

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "meta": {
                    "category": router_res.intent,
                    "agents": [router_res.route],
                    "citations": [],
                    "agent_messages": [{"intent": router_res.intent, "confidence": router_res.confidence}],
                    "status": "completed",
                    "elapsed": elapsed
                }
            })
