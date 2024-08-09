"""
File: nikki_tutor.py
Author: Scott Sullivan
Created: 2024-05-01
Description:
    This module is the main entry point for the Chatbot.

Functions:
    get_response(user_query, chat_history): returns a response to the user query
    main(): runs the chatbot
"""

# Variable Loaders
import sys, os, types
from dotenv import load_dotenv 

# Loads variables like API keys from the .env file
load_dotenv()
# os.environ["API_KEY"] = os.getenv("API_KEY")
# os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

# Ollama
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings

# Langchain
from langchain.schema.runnable import RunnablePassthrough, RunnableParallel, RunnableLambda
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma

# Prompts
# from langchain.prompts import PromptTemplate, MessagesPlaceholder, ChatPromptTemplate
from langchain_core.prompts import PromptTemplate, MessagesPlaceholder, ChatPromptTemplate
from langchain.chains.conversation.prompt import PROMPT

# Agents and Tools
from langchain.chains.question_answering import load_qa_chain
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser

from langchain.tools import BaseTool


# Streamlit
import streamlit as st
from streamlit_chat import message




# Local Imports
from rag.prompts.nikki_personality import nikki_prompt_template_writer
from constants import LANGUAGE_CHROMO_PATH, EMBED_MODEL




##
##  SETUP LLM  ##
##

# transformer_model = "gemma2"
# transformer_model = "gemma2:27b"
# transformer_model = "mixtral:8x7b"  #best for languange tutor so far

# transformer_model = "qwen2:7b"
# transformer_model = "llama3.1:70b"
transformer_model = "llama3.1"

llm = Ollama(model=transformer_model, temperature=0.5)

prompt = nikki_prompt_template_writer

# TODO: Add LoRA to the chain for Nikki's personality


## SETUP STREAMLIT APP ##
st.set_page_config(page_title="Nikki Writing Assistant")
st.title("Nikki Writing Assistant")

# Session State
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Hi! I'm Nikki, your writing assistant. What can I help you with?"),
    ]


def get_response(user_query, chat_history):
    embedding_function = SentenceTransformerEmbeddings(model_name=EMBED_MODEL)
    vectordb = Chroma(
        persist_directory=LANGUAGE_CHROMO_PATH,
        embedding_function=embedding_function
    )
    ollama_embeddings = OllamaEmbeddings(
        model=transformer_model,
        temperature=0.8
    )

    # history is limited to 25 messages
    formatted_history = "\n".join([f"{'Human' if isinstance(msg, HumanMessage) else 'AI'}: "for msg in chat_history[-20:]])  

    chain = (
        {
            "user_question": RunnablePassthrough(),
            "chat_history": lambda _: formatted_history
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    # Format user query and history for the chain
    input_data = {
        "user_question": user_query,
        "chat_history": formatted_history
    }
    response = chain.stream(input_data)
    return response



# # # Conversation
for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.write(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.write(message.content)


# User Input
user_query = st.chat_input("Type your message here...")
if user_query is not None and user_query != "":
    st.session_state.chat_history.append(HumanMessage(content=user_query))

    with st.chat_message("Human"):
        st.markdown(user_query)

    with st.chat_message("AI"):
        response = st.write_stream(get_response(user_query, st.session_state.chat_history))
        # response = get_response(user_query)
 
        # st.write(response)
    st.session_state.chat_history.append(AIMessage(content=response))





