import json
from pathlib import Path

import streamlit as st

from rag_persona_system import ConversationRAGSystem, PIPELINE_VERSION


st.set_page_config(page_title="Persona RAG Chatbot", page_icon="🤖", layout="wide")
st.title("Conversation Persona Chatbot")
st.caption("Uses topic checkpoints + 100-message checkpoints + message chunks + persona JSON")

ARTIFACT_DIR = Path("artifacts")
DATASET = Path("conversations.csv")


@st.cache_resource
def load_system() -> ConversationRAGSystem:
    system = ConversationRAGSystem(chunk_size=8, chunk_overlap=2, persona_speaker="User 1")
    if (ARTIFACT_DIR / "messages.csv").exists() and (ARTIFACT_DIR / "persona.json").exists():
        system.messages_df = __import__("pandas").read_csv(ARTIFACT_DIR / "messages.csv")
        system.topic_checkpoints = json.loads((ARTIFACT_DIR / "topic_checkpoints.json").read_text(encoding="utf-8"))
        system.hundred_checkpoints = json.loads((ARTIFACT_DIR / "message_checkpoints_100.json").read_text(encoding="utf-8"))
        system.chunks = json.loads((ARTIFACT_DIR / "message_chunks.json").read_text(encoding="utf-8"))
        system.persona = json.loads((ARTIFACT_DIR / "persona.json").read_text(encoding="utf-8"))
        persona_version = system.persona.get("meta", {}).get("pipeline_version")
        if persona_version == PIPELINE_VERSION:
            system.build_indices()
            return system

    if not DATASET.exists():
        raise FileNotFoundError("Missing conversations.csv. Run notebook first or add dataset.")
    system.run_pipeline(DATASET, ARTIFACT_DIR)
    return system


system = load_system()

col1, col2 = st.columns([2, 1])
with col1:
    st.subheader("Ask a question")
    default_questions = [
        "What kind of person is this user?",
        "What are their habits?",
        "How do they talk?",
    ]
    selected_prompt = st.selectbox("Quick prompts", default_questions, index=0)
    custom_q = st.text_input("Or type your own query", value="")
    q = custom_q.strip() or selected_prompt

    if st.button("Get answer", type="primary"):
        result = system.answer_query(q)
        st.markdown("### Answer")
        st.write(result["answer"])

        # Keep the UI concise by default; raw retrieval is still available for inspection.
        with st.expander("Show retrieval evidence (debug view)"):
            st.markdown("#### Top Summary Checkpoints")
            st.json(result["summary_hits"])
            st.markdown("#### Top Message Chunks")
            st.json(result["chunk_hits"])

with col2:
    st.subheader("Persona Snapshot")
    st.json(system.persona)
