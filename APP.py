import streamlit as st
import tempfile
import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough


st.set_page_config(page_title="Local RAG", layout="centered")
st.title("📄 Document Q&A using Local RAG (Ollama)")

uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file:
    with st.spinner("Processing PDF..."):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            pdf_path = tmp.name

        # Load PDF
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()

        # Split text
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=400,
            chunk_overlap=40
        )
        chunks = splitter.split_documents(documents)

        # Embeddings
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        # Vector store
        vectorstore = FAISS.from_documents(chunks, embeddings)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

        # Local LLM
        llm = Ollama(
            model="gemma:2b",
            temperature=0
        )

        # Prompt
        prompt = ChatPromptTemplate.from_template(
            """You are a helpful assistant.
Use ONLY the context below to answer.

Context:
{context}

Question:
{question}
"""
        )

        def format_docs(docs):
            return "\n\n".join(d.page_content for d in docs)

        # RAG chain (MODERN)
        rag_chain = (
            {
                "context": retriever | format_docs,
                "question": RunnablePassthrough(),
            }
            | prompt
            | llm
        )

    st.success("✅ PDF indexed successfully")

    question = st.text_input("Ask a question from the document")

    if question:
        with st.spinner("Generating answer..."):
            answer = rag_chain.invoke(question)
        st.write("### Answer")
        st.write(answer)

    os.remove(pdf_path)
