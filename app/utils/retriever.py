from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.retrievers import EnsembleRetriever
from langchain.schema import Document


# Simple hybrid retriever setup
def get_retriever(chunks: list[Document]):
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embedding_model)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    return retriever
