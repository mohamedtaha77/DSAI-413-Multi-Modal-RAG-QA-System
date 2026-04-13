# interface/app.py
# QA chatbot interface — upload a PDF, ask questions, get cited answers.

import os
import tempfile
import streamlit as st
from ingestion.embedder import ingest_pdf
from retrieval.retriever import Retriever
from generation.generator import generate_answer

st.set_page_config(page_title="RAG-Based QA", layout="wide")
st.title("Multi-Modal RAG QA System")

# --- Sidebar: PDF ingestion ---
with st.sidebar:
    st.header("Document Ingestion")
    uploaded = st.file_uploader("Upload a PDF", type=["pdf"])
    if uploaded and st.button("Ingest PDF"):
        with st.spinner("Ingesting…"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded.read())
                tmp_path = tmp.name
            ingest_pdf(tmp_path)
            os.unlink(tmp_path)
        st.success("Ingestion complete!")

# --- Main area: QA ---
query = st.text_input("Ask a question about the ingested document:")

if query:
    if "retriever" not in st.session_state:
        st.session_state.retriever = Retriever()

    with st.spinner("Retrieving and generating answer…"):
        pages = st.session_state.retriever.retrieve(query)
        answer = generate_answer(query, pages)

    st.subheader("Answer")
    st.write(answer)

    st.subheader("Retrieved Pages")
    cols = st.columns(len(pages))
    for col, page in zip(cols, pages):
        col.image(page.image, caption=f"{page.doc_name} — Page {page.page_num}")
