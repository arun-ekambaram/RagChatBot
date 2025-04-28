from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.retrievers.multi_query import ReciprocalRankFusionRetriever
from langchain.schema import Document
from langchain.retrievers import BM25Retriever, EnsembleRetriever

from typing import List

def get_retriever(chunks: List[Document]):
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embedding_model)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 8})


    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = 8

    
    hybrid_retriever = ReciprocalRankFusionRetriever(
        retrievers=[retriever, bm25_retriever],
        k=4
    )

    return hybrid_retriever
