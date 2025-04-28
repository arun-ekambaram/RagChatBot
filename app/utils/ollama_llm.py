from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

def get_rag_chain(retriever):
    template = """
    You are a helpful assistant. Use only the provided context to answer the question.
    If the answer is not in the context, say "The document does not contain that information."

    Context:
    {context}

    Question:
    {question}
    """
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=template,
    )

    llm = Ollama(model="llama3")  

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=False
    )
    return qa_chain
