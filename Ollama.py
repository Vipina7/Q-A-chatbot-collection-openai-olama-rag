import os
import streamlit as st
import openai
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama
from dotenv import load_dotenv

# Langchain Tracking
load_dotenv()
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Simple Q&A Chatbot With OLLAMA"

#Prompt Template
prompt = ChatPromptTemplate.from_messages([
    ("system","You are a helpful assistant. Please respond to the user queries."),
    ("user", "Question:{question}")
])

def get_response(question, llm, temperature):
    
    llm = Ollama(model = llm, temperature=temperature)
    output_parser = StrOutputParser()
    chain = prompt|llm|output_parser
    response = chain.invoke({"question":question})

    return response

## Streamlit App
st.title("Enhanced Q&A Chatbot With OLLAMA")

## Sidebar for settings
st.sidebar.title("Settings")

## Drop down to select various Open AI models
llm = st.sidebar.selectbox("Select an Open AI Model", ["gemma3:latest","gemma:2b","deepseek-r1:1.5b"])

## Adjust response parameter
temperature = st.sidebar.slider("Temperature",min_value=0.0, max_value=1.0, value=0.7)

## Main interface for user input
st.write("Go ahead and ask any question")
user_input = st.text_input('You:')

if user_input:
    response = get_response(question=user_input, llm=llm, temperature=temperature)
    st.write(response)
else:
    st.write("Please provide a query...")

