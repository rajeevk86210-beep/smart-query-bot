"""Local RAG Smart Grievance and Query Resolution Bot."""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

PERSIST_DIR = Path(__file__).resolve().parent / "chroma_db"
COLLECTION_NAME = "grievance_knowledge"
EMBEDDING_MODEL = "gemini-embedding-2-preview"
CHAT_MODEL = "gemini-3.5-flash"

SYSTEM_PROMPT = """You are a Smart Grievance and Query Resolution assistant for staff and citizens.
Use ONLY the retrieved document context (rulebooks, FAQs, policies) to answer.

Guidelines:
- Be professional, concise, and empathetic.
- If the user describes a grievance, acknowledge it, map it to relevant rules in the context, and outline practical next steps.
- Quote or paraphrase the applicable rule when the context supports it, and mention the source file/page if available.
- If the context is insufficient, say so clearly. Do not invent policies, deadlines, or contact details.
"""

QA_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        (
            "human",
            "Context from official documents:\n{context}\n\n"
            "User query or grievance:\n{question}",
        ),
    ]
)


def resolve_api_key() -> str:
    sidebar_key = st.session_state.get("gemini_api_key_input", "").strip()
    if sidebar_key:
        return sidebar_key
    return (
        os.getenv("GEMINI_API_KEY", "").strip()
        or os.getenv("GOOGLE_API_KEY", "").strip()
    )


def apply_api_key(api_key: str) -> None:
    os.environ["GEMINI_API_KEY"] = api_key
    os.environ["GOOGLE_API_KEY"] = api_key


def get_embeddings(api_key: str) -> GoogleGenerativeAIEmbeddings:
    return GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        google_api_key=api_key,
    )


def format_docs(docs) -> str:
    parts = []
    for i, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page")
        page_label = f"page {page + 1}" if isinstance(page, int) else "page n/a"
        parts.append(f"[{i}] ({Path(str(source)).name}, {page_label})\n{doc.page_content}")
    return "\n\n".join(parts) if parts else "No relevant documents were retrieved."


def load_vectorstore(api_key: str) -> Chroma:
    PERSIST_DIR.mkdir(parents=True, exist_ok=True)
    return Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=str(PERSIST_DIR),
        embedding_function=get_embeddings(api_key),
    )


def ingest_pdfs(uploaded_files, api_key: str) -> tuple[int, int]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""],
    )

    documents = []
    with tempfile.TemporaryDirectory() as tmp_dir:
        for uploaded in uploaded_files:
            tmp_path = Path(tmp_dir) / uploaded.name
            tmp_path.write_bytes(uploaded.getvalue())
            loader = PyPDFLoader(str(tmp_path))
            pages = loader.load()
            for page in pages:
                page.metadata["source"] = uploaded.name
            documents.extend(pages)

    chunks = splitter.split_documents(documents)
    if not chunks:
        raise ValueError("No text could be extracted from the uploaded PDF(s).")

    vectorstore = load_vectorstore(api_key)
    vectorstore.add_documents(chunks)
    if hasattr(vectorstore, "persist"):
        vectorstore.persist()
    return len(documents), len(chunks)


def reset_knowledge_base() -> None:
    st.session_state.pop("vectorstore", None)
    if PERSIST_DIR.exists():
        shutil.rmtree(PERSIST_DIR)


def build_rag_chain(vectorstore: Chroma, api_key: str):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    llm = ChatGoogleGenerativeAI(
        model=CHAT_MODEL,
        google_api_key=api_key,
        temperature=0.2,
        streaming=True,
    
    )
    return (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | QA_PROMPT
        | llm
        | StrOutputParser()
    )


def render_sidebar() -> None:
    st.sidebar.header("Configuration")
    if "gemini_api_key_input" not in st.session_state:
        st.session_state.gemini_api_key_input = (
            os.getenv("GEMINI_API_KEY", "") or os.getenv("GOOGLE_API_KEY", "")
        )
    st.sidebar.text_input(
        "GEMINI_API_KEY",
        type="password",
        key="gemini_api_key_input",
        help="Paste a Gemini API key here, or set GEMINI_API_KEY in a local .env file.",
    )

    api_key = resolve_api_key()
    if api_key:
        apply_api_key(api_key)
        st.sidebar.success("API key loaded")
    else:
        st.sidebar.warning("Add a Gemini API key to enable indexing and chat.")

    st.sidebar.divider()
    st.sidebar.header("Knowledge base")
    uploaded_files = st.sidebar.file_uploader(
        "Upload PDF rulebooks or FAQs",
        type=["pdf"],
        accept_multiple_files=True,
        help="Documents are chunked, embedded with Gemini embedding-004, and stored in local ChromaDB.",
    )

    col_index, col_reset = st.sidebar.columns(2)
    with col_index:
        process_clicked = st.button("Process PDFs", use_container_width=True, type="primary")
    with col_reset:
        reset_clicked = st.button("Reset store", use_container_width=True)

    if reset_clicked:
        reset_knowledge_base()
        st.sidebar.info("Local ChromaDB store cleared.")

    if process_clicked:
        if not api_key:
            st.sidebar.error("Set GEMINI_API_KEY before processing PDFs.")
        elif not uploaded_files:
            st.sidebar.error("Upload at least one PDF.")
        else:
            with st.sidebar.status("Indexing documents…", expanded=True) as status:
                try:
                    pages, chunks = ingest_pdfs(uploaded_files, api_key)
                    st.session_state.vectorstore = load_vectorstore(api_key)
                    status.update(
                        label=f"Indexed {pages} pages into {chunks} chunks.",
                        state="complete",
                    )
                except Exception as exc:
                    status.update(label="Indexing failed", state="error")
                    st.sidebar.error(str(exc))

    persist_exists = PERSIST_DIR.exists() and any(PERSIST_DIR.iterdir())
    st.sidebar.caption(
        f"Vector store: {'ready at chroma_db/' if persist_exists else 'empty — upload PDFs to begin'}."
    )


def main() -> None:
    st.set_page_config(
        page_title="Smart Grievance & Query Bot",
        page_icon="⚖️",
        layout="wide",
    )
    st.title("Smart Grievance and Query Resolution Bot")
    st.caption(
        "Ask questions or describe a grievance. Answers are grounded in your uploaded PDFs "
        "via ChromaDB retrieval and Gemini 2.5 Flash."
    )

    render_sidebar()
    api_key = resolve_api_key()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    persist_exists = PERSIST_DIR.exists() and any(PERSIST_DIR.iterdir())
    if persist_exists and api_key and "vectorstore" not in st.session_state:
        try:
            st.session_state.vectorstore = load_vectorstore(api_key)
        except Exception:
            st.session_state.pop("vectorstore", None)

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    prompt = st.chat_input("Ask a question or describe your grievance…")
    if not prompt:
        return

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    if not api_key:
        reply = "Please set a GEMINI_API_KEY in the sidebar or in a `.env` file, then try again."
        with st.chat_message("assistant"):
            st.markdown(reply)
        st.session_state.messages.append({"role": "assistant", "content": reply})
        return

    vectorstore = st.session_state.get("vectorstore")
    if vectorstore is None:
        reply = "Upload and process at least one PDF in the sidebar so I can retrieve official context."
        with st.chat_message("assistant"):
            st.markdown(reply)
        st.session_state.messages.append({"role": "assistant", "content": reply})
        return

    with st.chat_message("assistant"):
        with st.spinner("Retrieving context and drafting a response…"):
            try:
                chain = build_rag_chain(vectorstore, api_key)
                answer = st.write_stream(chain.stream(prompt))
            except Exception as exc:
                answer = f"I could not complete that request: {exc}"
        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()
