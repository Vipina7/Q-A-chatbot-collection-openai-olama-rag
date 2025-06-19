import os
import openai
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

import streamlit as st

# Set the environmental variables
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_PROJECT"] = "Simple Q&A Chatbot With OpenAI"
os.environ["LANGCHAIN_TRACING_V2"] = "true"

## Chat Prompt
prompt = ChatPromptTemplate.from_messages([
    ("system","You are an helpful AI Assistant that answers the questions of users."),
    ("user","{question}")
])

## Function to get the openai response
def get_response(question, llm, temperature, max_tokens, api_key):
    llm = ChatOpenAI(api_key=api_key, model=llm, temperature=temperature, max_tokens = max_tokens)
    output_parser = StrOutputParser()
    chain = prompt|llm|output_parser
    response = chain.invoke({"question":question})

    return response

## Streamlit App
st.title("Enhanced Q&A Chatbot With OpenAI")

## sidebar for settings
st.sidebar.title("Settings")

api_key = st.sidebar.text_input("Enter your OpenAI API Key", type="password")
model = st.sidebar.selectbox("Select an OpenAI model", ["gpt-4o","gpt-4-turbo","gpt-4"])

temperature = st.sidebar.slider("Temperatute", min_value=0.0, max_value=1.0, value=0.7)
max_tokens = st.sidebar.slider("Max Tokens", min_value=50, max_value=300, value=150)

## Main interface for user input
st.write("Go ahead and ask any question")
user_input = st.text_input("You:")

if user_input:
    response = get_response(question=user_input, llm=model, temperature=temperature, max_tokens=max_tokens, api_key=api_key)
    st.write(response)
else:
    st.write("Please provide a query...")

    