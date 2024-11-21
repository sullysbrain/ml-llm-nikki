"""
File: nikki_generic_ai.py
Author: Scott Sullivan
Created: 2024-05-01
Description:
    This module is the main entry point for the Chatbot.

Functions:
    get_response(user_query, chat_history): returns a response to the user query
    main(): runs the chatbot
"""


MAX_CHAT_HISTORY = 25

# Variable Loaders
import sys, os, types
from dotenv import load_dotenv 

# Loads variables like API keys from the .env file
load_dotenv()

# Llama.cpp
from langchain_community.llms import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_community.embeddings import LlamaCppEmbeddings

# Ollama
# from langchain_community.llms import Ollama
# from langchain_community.embeddings import OllamaEmbeddings

# Langchain
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_chroma import Chroma
from langchain.schema.runnable import RunnablePassthrough, RunnableParallel, RunnableLambda

from langchain_huggingface import HuggingFaceEmbeddings


# Prompts
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
import rag.prompts.nikki_personality as nikki_personality
from constants import EMBED_MODEL, LANGUAGE_CHROMADB_PATH


# Helper functions
def format_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])

def format_metadata(docs):
    return "\n".join([str(d.metadata) for d in docs])

def format_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])



##
##  SETUP LLM  ##
##


###### Llama CPP

# model_path="./models/Llama-3.2-3B-Instruct-uncensored-Q4_K_M.gguf"
# model_path="./models/gemma-7b-it-Q8_0.gguf"
# model_path="./models/Gemma-The-Writer-N-Restless-Quill-10B-D_AU-Q4_k_s.gguf"
model_path="./models/qwen2.5-3b-instruct-q8_0.gguf"

callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
llm = LlamaCpp(
    model_path=model_path,
    temperature=0.3,  # Lower temperature for more focused output
    n_ctx=131072,  # Adjust to an optimal context size (8192 for handling large chunks)
    n_ctx_per_seq=131072,
    n_batch=32,  # Adjust batch size to fit within memory limits (e.g., 8-16 tokens)
    max_tokens=512,  # Adjust max tokens for concise responses
    top_p=0.9,  # Use nucleus sampling for more controlled output
    callback_manager=callback_manager,
    verbose=False
)

llama_embeddings = LlamaCppEmbeddings(
    model_path=model_path,
    n_ctx=2048,
    n_batch=16  # use 8 if too slow
)


# Prompt
prompt = nikki_personality.nikki_writer_gemma



# TODO: Add LoRA to the chain for Nikki's personality
# Use the Colab notebook to generate the LoRA from json examples

## SETUP STREAMLIT APP ##
st.set_page_config(page_title="Writer Assistant")
st.title("Writer Assistant Chatbot")

# Session State
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Hi! What can I help you with?"),
    ]




### MAIN LLM FUNCTION ###

def get_response(user_query, chat_history):

    # history is limited to 25 messages
    formatted_history = "\n".join([f"{'Human' if isinstance(msg, HumanMessage) else 'AI'}: {msg.content}" for msg in chat_history[-25:]])  

    # Prepare the chain
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

    if len(st.session_state.chat_history) > MAX_CHAT_HISTORY:
        st.session_state.chat_history.pop(0)


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





