import streamlit as st
from loader import extract_video_id
from chunking import getTranscriptChunks
from summarizer import summarize_chunks, summarize_all
from vector_store import YouTubeVectorStore
from qa import answer
from langchain_core.messages import HumanMessage, AIMessage
import sys
import types

sys.modules['torch.classes'] = types.ModuleType('torch.classes')

st.set_page_config(page_title="🎥 YouTube QA Chat", layout="wide")
st.title("📽️ YouTube Video Summarizer + QA Chat")

if "summary_ready" not in st.session_state:
    st.session_state.summary_ready = False
    st.session_state.final_summary = ""
    st.session_state.retriever = None
    st.session_state.chat_history = []

video_url = st.text_input("Enter a YouTube video URL:", "")

if video_url:
    video_id = extract_video_id(video_url)
    store = YouTubeVectorStore(video_id)

    if store.vectorstore and not st.session_state.summary_ready:
        summary_chunk_list = store.get_chunks()
        st.session_state.final_summary = summarize_all(summary_chunk_list)
        st.session_state.retriever = store.get_multiquery_retriever()
        st.session_state.summary_ready = True

    elif not store.vectorstore and not st.session_state.summary_ready:
        with st.spinner("Processing video and generating summary..."):
            transcript_chunks = getTranscriptChunks(video_url)
            summary_chunks = summarize_chunks(transcript_chunks)
            store.create_faiss_store(transcript_chunks, summary_chunks)
            st.session_state.final_summary = summarize_all(summary_chunks)
            st.session_state.retriever = store.get_multiquery_retriever()
            st.session_state.summary_ready = True
            st.success("✅ Video processed and stored!")

if st.session_state.summary_ready:
    with st.expander("📄 Summary (click to expand/collapse)", expanded=True):
        st.markdown(
            f"<div style='font-size: 20px; line-height: 1.6'>{st.session_state.final_summary}</div>",
            unsafe_allow_html=True
        )

    st.divider()
    st.subheader("🤖 Chat about this video")

    col1, col2 = st.columns([5, 3])
    with col1:
        st.markdown("### 💬 Chat History")
        if st.session_state.chat_history:
            for msg in st.session_state.chat_history:
                role = "👤 You" if isinstance(msg, HumanMessage) else "🤖 AI"
                st.markdown(
                f"<div style='font-size: 18px; margin-bottom: 10px'><strong>{role}:</strong> {msg.content}</div>",
                unsafe_allow_html=True
            )
        else:
            st.info("Ask a question about the video to start the chat.")

    with col2:
        with st.form("chat_form", clear_on_submit=True):
            user_question = st.text_input("Your question", key="user_input")
            submitted = st.form_submit_button("Ask")

        if submitted and user_question:
            st.session_state.chat_history.append(HumanMessage(content=user_question))
            response = answer(user_question, st.session_state.chat_history, st.session_state.retriever)
            st.session_state.chat_history.append(response)
            st.rerun() 
