import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
import streamlit as st

## Set environmental variables
load_dotenv()
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
os.environ["LANGCHAIN_PROJECT"] = "RAG Q&A Chatbot With GROQ"
os.environ["LANGCHAIN_TRACING_V2"] = "true"

## LLM model
llm = ChatGroq(model="Llama3-8b-8192")

## Prompt
prompt = ChatPromptTemplate.from_template(
    """
    Answer the question based on the provided context only.
    Please provide the most accurate response based on the question.
    <context>
    {context}
    <context>
    Question:{input}
    """
    
)

# Get vector embeddings
def create_vector_embeddings():
    if 'vectors' not in st.session_state:
        st.session_state.embeddings = HuggingFaceEmbeddings()
        st.session_state.loader = PyPDFDirectoryLoader("repository")
        st.session_state.docs = st.session_state.loader.load()
        st.session_state.splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        st.session_state.splits = st.session_state.splitter.split_documents(st.session_state.docs)
        st.session_state.vectors = FAISS.from_documents(st.session_state.splits, st.session_state.embeddings)

## Stramlit User Interface
st.title('RAG Document Q&A With Groq And Llama3')
user_input = st.text_input("Enter you query for the research paper")

if st.button("Document Embedding"):
    create_vector_embeddings()
    st.write("Vector Database is ready")

import time

if user_input:
    document_chain = create_stuff_documents_chain(llm,prompt)
    retriever = st.session_state.vectors.as_retriever()
    rag_chain = create_retrieval_chain(retriever,document_chain)

    start = time.process_time()
    response = rag_chain.invoke({"input":user_input})
    print(f"Response time: {time.process_time() - start}")

    st.write(response["answer"])

    ## With a streamlit expander
    with st.expander("Document Similarity Search"):
        for i,doc in enumerate(response["context"]):
            st.write(doc.page_content)
            st.write('------------------------------------')