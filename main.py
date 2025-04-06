import streamlit as st
from app.utils.loaders import load_and_chunk_document
from app.utils.retriever import get_retriever
from app.utils.ollama_llm import get_rag_chain

st.set_page_config(page_title="RAG Document Chatbot", layout="wide")
st.title("RAG-based Document Chatbot (Ollama + LLaMA 3)")

uploaded_file = st.file_uploader("Upload a PDF Document", type=["pdf"])

if uploaded_file:
    with st.spinner("Processing document..."):
        chunks = load_and_chunk_document(uploaded_file)
        retriever = get_retriever(chunks)
        rag_chain = get_rag_chain(retriever)
    st.success("Document processed. Ask your question below.")

    query = st.text_input("Ask a question about the document")
    if query:
        with st.spinner("Generating response..."):
            response = rag_chain.invoke({"query": query})
            st.markdown("---")
            st.markdown("### Answer:")
            st.markdown(response["result"])


    
