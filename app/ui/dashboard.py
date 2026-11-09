import os
import sys
from pathlib import Path

# Add project root to sys.path so 'app' package is always found
project_root = str(Path(__file__).resolve().parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Remove any shadowed 'app' module
if "app" in sys.modules and not hasattr(sys.modules["app"], "__path__"):
    del sys.modules["app"]

import streamlit as st
import time
import uuid

from app.agents.graph import run_agent_workflow
from app.router.engine import default_router
from app.router.models import RouterRequest
from app.memory.conversation import conversation_manager
from app.memory.manager import memory_manager
from app.memory.extractor import memory_extractor
from app.memory.schemas import MemoryCategoryEnum
from app.database import init_db
from app.config import settings

# Initialize database
init_db()

st.set_page_config(
    page_title="Multi-Agent RAG Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .badge-agent {
        display: inline-block;
        padding: 3px 9px;
        border-radius: 12px;
        font-size: 0.76rem;
        font-weight: 600;
        background-color: #ede9fe;
        color: #5b21b6;
        margin-right: 5px;
        margin-bottom: 3px;
    }
    .badge-category {
        display: inline-block;
        padding: 3px 9px;
        border-radius: 12px;
        font-size: 0.76rem;
        font-weight: 600;
        background-color: #e0f2fe;
        color: #0369a1;
        margin-right: 5px;
        margin-bottom: 3px;
    }
    .badge-status {
        display: inline-block;
        padding: 3px 9px;
        border-radius: 12px;
        font-size: 0.76rem;
        font-weight: 600;
        background-color: #ecfdf5;
        color: #047857;
        margin-bottom: 3px;
    }
    .badge-memory {
        display: inline-block;
        padding: 3px 9px;
        border-radius: 12px;
        font-size: 0.76rem;
        font-weight: 600;
        background-color: #fef3c7;
        color: #92400e;
        margin-right: 5px;
        margin-bottom: 3px;
    }
</style>
""", unsafe_allow_html=True)


# Initialize Session State
if "user_id" not in st.session_state:
    st.session_state.user_id = "default_user"

if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = f"session_{uuid.uuid4().hex[:8]}"

if "messages" not in st.session_state:
    st.session_state.messages = []


# Helper function to reload messages from DB
def reload_messages_from_db():
    conv_data = conversation_manager.get_conversation(st.session_state.conversation_id, st.session_state.user_id)
    if conv_data and conv_data.get("messages"):
        reloaded = []
        for m in conv_data["messages"]:
            meta = m.get("metadata", {})
            reloaded.append({
                "role": m["role"],
                "content": m["content"],
                "meta": {
                    "category": meta.get("category", ""),
                    "agents": meta.get("agents_used", []),
                    "citations": meta.get("citations", []),
                    "agent_messages": []
                }
            })
        st.session_state.messages = reloaded
    else:
        st.session_state.messages = []


# Sidebar: Conversations & Memory Management
with st.sidebar:
    st.title("🤖 Assistant Control")
    st.caption("LangGraph Multi-Agent with 3-Tier Memory")

    # User Profile Switcher
    st.markdown("---")
    user_input = st.text_input("User Identity", value=st.session_state.user_id)
    if user_input != st.session_state.user_id:
        st.session_state.user_id = user_input.strip() or "default_user"
        st.session_state.conversation_id = f"session_{uuid.uuid4().hex[:8]}"
        st.session_state.messages = []
        st.rerun()

    # Conversation History Management
    st.markdown("---")
    st.subheader("💬 Conversations")
    
    col_new, col_del = st.columns([1, 1])
    with col_new:
        if st.button("➕ New Chat", use_container_width=True):
            st.session_state.conversation_id = f"session_{uuid.uuid4().hex[:8]}"
            st.session_state.messages = []
            st.rerun()
    with col_del:
        if st.button("🗑️ Delete Chat", use_container_width=True):
            conversation_manager.delete_conversation(st.session_state.conversation_id, st.session_state.user_id)
            st.session_state.conversation_id = f"session_{uuid.uuid4().hex[:8]}"
            st.session_state.messages = []
            st.rerun()

    # List past conversations
    past_convs = conversation_manager.list_conversations(st.session_state.user_id, limit=20)
    conv_options = {c["id"]: f"{c['title']} ({c['message_count']} msgs)" for c in past_convs}
    
    if st.session_state.conversation_id not in conv_options:
        conv_options[st.session_state.conversation_id] = "Current Active Session"

    selected_cid = st.selectbox(
        "Select Active Conversation",
        options=list(conv_options.keys()),
        index=list(conv_options.keys()).index(st.session_state.conversation_id),
        format_func=lambda x: conv_options.get(x, x)
    )

    if selected_cid != st.session_state.conversation_id:
        st.session_state.conversation_id = selected_cid
        reload_messages_from_db()
        st.rerun()

    # Long-Term Memory Manager
    st.markdown("---")
    with st.expander("🧠 Persistent Long-Term Memories", expanded=False):
        st.caption("Memories recalled across distinct conversations.")
        
        # Add new memory form
        with st.form("add_memory_form"):
            st.markdown("**Add Explicit Memory**")
            new_cat = st.selectbox(
                "Category",
                [c.value for c in MemoryCategoryEnum]
            )
            new_content = st.text_area("Memory Content", placeholder="e.g. User prefers Python 3.13 with type annotations.")
            new_imp = st.slider("Importance", 1, 5, 3)
            submitted = st.form_submit_button("Save Memory")
            if submitted and new_content:
                res = memory_manager.save_memory(
                    user_id=st.session_state.user_id,
                    content=new_content,
                    category=new_cat,
                    importance=new_imp
                )
                if res:
                    st.success("Memory saved!")
                    st.rerun()
                else:
                    st.error("Invalid memory or contains sensitive secrets.")

        # List existing memories
        user_memories = memory_manager.get_user_memories(st.session_state.user_id)
        if user_memories:
            for mem in user_memories:
                c_del1, c_del2 = st.columns([4, 1])
                with c_del1:
                    st.markdown(f"- **[{mem['category'].replace('_', ' ').title()}]**: {mem['content']} *(Imp: {mem['importance']})*")
                with c_del2:
                    if st.button("❌", key=f"del_{mem['id']}"):
                        memory_manager.delete_memory(mem['id'], st.session_state.user_id)
                        st.rerun()
        else:
            st.info("No long-term memories saved yet.")

    st.markdown("---")
    st.subheader("Orchestration Mode")
    engine_mode = st.radio(
        "Execution Engine",
        ["LangGraph Multi-Agent (6 Agents)", "Direct Query Router (Single-Turn)"],
        index=0
    )
    show_agent_trace = st.toggle("Show Agent Execution Logs", value=True)


# Main Interface Header
st.title("🤖 Multi-Agent RAG Assistant")
st.caption(f"Session: `{st.session_state.conversation_id}` | User: `{st.session_state.user_id}` | Orchestrated with LangGraph & 3-Tier Memory")

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

            badges_html = '<div style="margin-top: 8px; margin-bottom: 6px;">'
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
if prompt := st.chat_input("Ask a research question, follow up on previous context, or request a calculation..."):
    # Append user prompt
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Execute Assistant
    with st.chat_message("assistant"):
        start_time = time.time()
        
        if "LangGraph" in engine_mode:
            with st.spinner("Multi-Agent Graph is reasoning with short-term context & long-term memory..."):
                final_state = run_agent_workflow(
                    query=prompt,
                    conversation_id=st.session_state.conversation_id,
                    user_id=st.session_state.user_id
                )
            
            elapsed = time.time() - start_time
            answer = final_state.get("final_response", "No response generated.")
            category = final_state.get("query_category", "UNKNOWN")
            agents = final_state.get("selected_agents", [])
            citations = final_state.get("source_references", [])
            messages = final_state.get("agent_messages", [])
            status = final_state.get("execution_status", "completed")

            st.markdown(answer)

            # Persist to database history
            conversation_manager.add_turn(
                conversation_id=st.session_state.conversation_id,
                user_id=st.session_state.user_id,
                query=prompt,
                response=answer,
                category=category,
                agents_used=agents,
                citations=citations
            )

            # Trigger selective memory extraction
            try:
                extracted = memory_extractor.extract_and_save(
                    user_id=st.session_state.user_id,
                    query=prompt,
                    response=answer,
                    conversation_id=st.session_state.conversation_id
                )
            except Exception:
                extracted = []

            # Metadata Badges
            badges_html = '<div style="margin-top: 8px; margin-bottom: 6px;">'
            badges_html += f'<span class="badge-category">Category: {category}</span>'
            for a in agents:
                badges_html += f'<span class="badge-agent">🤖 {a}</span>'
            if extracted:
                badges_html += f'<span class="badge-memory">🧠 Saved {len(extracted)} Memory</span>'
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
