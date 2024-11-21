from langchain import hub
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import CharacterTextSplitter
from langchain_ollama import OllamaLLM, OllamaEmbeddings


if __name__ == "__main__":
    loader = PyPDFLoader(file_path="react.pdf")
    documents = loader.load()

    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=30, separator="\n")
    document_data = splitter.split_documents(documents)

    embeddings = OllamaEmbeddings(model="llama3.1")
    # vectorstore = FAISS.from_documents(document_data, embeddings)
    # vectorstore.save_local("vectorstore")
    vectorstore = FAISS.load_local("vectorstore", embeddings, allow_dangerous_deserialization=True)

    llm = OllamaLLM(model="llama3.1")
    prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    combine_docs_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(vectorstore.as_retriever(), combine_docs_chain)

    res = retrieval_chain.invoke({"input": "Give me the gist of ReAct in 3 sentences"})

    print(res["answer"])
