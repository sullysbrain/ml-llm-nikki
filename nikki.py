"""
File: nikki.py
Author: Scott Sullivan
Created: 2024-05-01
Description:
    This module is the main entry point for the Chatbot.

Functions:
    get_response(user_query, chat_history): returns a response to the user query
    main(): runs the chatbot
"""

# Variable Loaders
import dotenv, file_helper, os
dotenv.load_dotenv()

# Import Langchain Tool & Agents
from langchain.agents import load_tools
from langchain.utilities import TextRequestsWrapper
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain_core.tools import BaseTool
import requests


# Vector Store
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema.runnable import RunnablePassthrough, RunnableParallel, RunnableLambda

# LLM - Ollama
from langchain_community.llms import Ollama
from langchain_experimental.llms.ollama_functions import OllamaFunctions
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# Prompts
from langchain.prompts import PromptTemplate
from langchain.chains.conversation.prompt import PROMPT
from langchain.chains import create_retrieval_chain
from langchain.chains import ConversationChain, ConversationalRetrievalChain
from langchain.chains.combine_documents import create_stuff_documents_chain

from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser

# Local Imports
import build_chain
import nikki_personality as nikki
import _private.template_ae as ae_chat

# Streamlit
import streamlit as st
from constants import REPORTS_CHROMA_PATH, EMBED_MODEL, LANGUAGE_CHROMO_PATH

st.set_page_config(page_title="Chatbot")
st.title("Chatbot")

# Initialize the Transformer
# Potential options: "llama3:8b", "llama2:13b", "llama3:8b", "mixtral:8x7b", "qwen:32b"
llm = build_chain.build_llm(transformer_name="mixtral:8x7b")


def get_response(user_query, chat_history):
    prompt = build_chain.build_prompts(ae_chat.ae_prompt_template)
    # prompt = build_chain.build_prompts(nikki.nikki_tutor_prompt_template)

    # RAG Constructor Arguments: vector embedded model, vector store path    
    retriever = build_chain.build_rag(model_name=EMBED_MODEL, database_directory=REPORTS_CHROMA_PATH)
    chain = (
        {"context": retriever, "user_question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain.stream(user_query)

    # return chain.stream({
    #     # "chat_history": chat_history,
    #     "user_question": user_query,
    # })

    

# Session State
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Hi! How can I help you?"),
    ]
    
# # Conversation
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

    st.session_state.chat_history.append(AIMessage(content=response))





