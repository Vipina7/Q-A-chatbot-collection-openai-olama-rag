import os
import streamlit as st
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

from dotenv import load_dotenv
load_dotenv()

os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")
embeddings = HuggingFaceEmbeddings(model_name = "all-MiniLM-L6-v2")

## Streamlit app
st.title("Conversational RAG wit PDF uploads and chat history")
st.write("Upload PDF's and chat with their content")

api_key = st.text_input("Enter your Groq API Key:", type="password")

if api_key:
    llm = ChatGroq(groq_api_key = api_key, model="Gemma2-9b-It")

    ## Chat interface
    
    session_id = st.text_input("Session_id", value="default_session_id")
    ## statefully manage chat history
    
    if 'store' not in st.session_state:
        st.session_state.store = {}
    
    uploaded_files = st.file_uploader("Upload your PDF file(S)", type="pdf", accept_multiple_files=True)

    ## Process uploaded files
    if uploaded_files:
        documents = []

        for uploaded_file in uploaded_files:
            temppdf = f"./temp.pdf"
            with open(temppdf,"wb") as file:
                file.write(uploaded_file.getvalue())
                file_name = uploaded_file.name

            loader = PyPDFLoader(temppdf)
            docs = loader.load()
            documents.extend(docs)

        ## SPlit and create embeddings
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(documents)
        vectorstore = Chroma.from_documents(splits,embeddings)
        retriever = vectorstore.as_retriever()

        contextualize_qa_system_prompt = (
            """
            Given a chat history and the latest user question which might 
            reference contect in the chat history, formulate a standalone
            question which can be understood without the chat history. Do NOT
            answer the question, just reformulate it if needed and otherwise return it as it is.
            """
            
        )

        contextualize_qa_prompt = ChatPromptTemplate.from_messages([
            ("system",contextualize_qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("user","{input}")
        ])

        history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_qa_prompt)

        ## Answer the question
        system_prompt = (
            """
            You are an assistant for question-answer tasks.
            Use the following pieces of retrieved context to answer
            the question. If you don't know the answer, say that you
            don't know. Use three sentences maximum and keep the answer concise.
            \n\n
            {context}
            """
            
        )

        qa_prompt = ChatPromptTemplate.from_messages([
            ("system",system_prompt),
            MessagesPlaceholder("chat_history"),
            ("user","{input}")
        ])

        question_answer_chain = create_stuff_documents_chain(llm,qa_prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

        ## Get session id
        def get_session_id(session_id:str) -> BaseChatMessageHistory:
            if session_id not in st.session_state.store:
                st.session_state.store[session_id] = ChatMessageHistory()
            
            return st.session_state.store[session_id]
        
        conversational_chain = RunnableWithMessageHistory(
            rag_chain, get_session_id, input_messages_key="input", history_messages_key="chat_history", output_messages_key="answer"
        )

        user_input = st.text_input("Your Question:")

        if user_input:
            session_history = get_session_id(session_id)
            response = conversational_chain.invoke(
                {"input":user_input},
                config={
                    "configurable":{"session_id":session_id}
                }
            )
            st.write(st.session_state.store)
            st.write("Assistant:", response["answer"])
            st.write("Chat History:", session_history.messages)

else:
    st.warning("Please enter the Groq API Key")
        





