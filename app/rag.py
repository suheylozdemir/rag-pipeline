from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
import os

load_dotenv()  # load API key from .env file

def create_rag_pipeline(file_path: str):
    # Step 1: Load document
    loader = TextLoader(file_path)
    documents = loader.load()

    # Step 2: Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,    # each chunk max 500 characters
        chunk_overlap=50   # 50 character overlap between chunks
    )
    chunks = splitter.split_documents(documents)

    # Step 3: Create embeddings and store in ChromaDB
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_documents(chunks, embeddings)

    # Step 4: Create retriever and QA chain
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

    return qa_chain


def ask_question(qa_chain, question: str):
    result = qa_chain.invoke({"query": question})
    return result["result"]