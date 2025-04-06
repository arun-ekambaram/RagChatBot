from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate


def get_rag_chain(retriever):
    template = """
    You are a helpful assistant. Answer only using the information from the document.
    If the answer is not present, say "The document does not contain that information."

    Question: {question}
    """
    prompt = PromptTemplate.from_template(template)

    llm = Ollama(model="llama3")  

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=False
    )
    return qa_chain
