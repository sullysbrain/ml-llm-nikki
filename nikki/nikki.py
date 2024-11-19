"""
File: nikki.py
Author: Scott Sullivan
Description:
    This module is the main entry point for the Chatbot.

Functions:
    get_response(user_query, chat_history): returns a response to the user query
    main(): runs the chatbot
"""

MAX_MESSAGES = 5


# Variable Loaders
import sys, os, types
from dotenv import load_dotenv 

# Loads variables like API keys from the .env file
load_dotenv()

cwd = os.getcwd()
print("Current working directory:", cwd)


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
from langchain_community.vectorstores import Chroma
from langchain.schema.runnable import RunnablePassthrough, RunnableParallel, RunnableLambda

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








# Prompts
from langchain.prompts import PromptTemplate
from langchain.chains.conversation.prompt import PROMPT

# Local Imports
import db_llm_builder as llm_builder
import rag.prompts.nikki_personality as nikki


from constants import MARKDOWN_REPORTS_PATH, ESTOP_LOG_PATH, REPORTS_CHROMA_PATH, EMBED_MODEL
import dotenv, file_helper, os
dotenv.load_dotenv()

from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler


###### Ollama
# from langchain_community.llms import Ollama

# original
# transformer_name = "mixtral:8x7b"
# transformer_name = "nsheth/llama-3-lumimaid-8b-v0.1-iq-imatrix"
# transformer_name = "qwen2:7b"

# llm = build_llm(transformer_name=transformer_name)

# llm = Ollama(
#     model=transformer_name, 
#     callback_manager=callback_manager,
#     stop=["<|start_header_id|>", 
#             "<|end_header_id|>", 
#             "<|eot_id|>", 
#             "<|reserved_special_token"]
# )


###### Llama CPP

# model_path="./models/gemma-7b-it-Q8_0.gguf"
# model_path="./models/llama_3_1_nikki_unsloth.Q4_K_M.gguf"
# model_path="./models/Meta-Llama-3_1-8B-Instruct-Q2_K_L.gguf"
# model_path="./models/Qwen2.5-32B-Instruct-Q4_K_S.gguf"
# model_path="./models/Qwen2.5-32B-Instruct-Q2_K.gguf"


# model_path="./models/Meta-Llama-3_1-8B-Instruct-Q3_K_L.gguf"
model_path="./models/Llama-3.2-3B-Instruct-uncensored-Q4_K_M.gguf"
# model_path="./models/Llama-3.2-3B-Instruct-Q6_K.gguf"

callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
llm = LlamaCpp(
    # Core parameters
    model_path=model_path,  # Path to your model
    n_ctx=4096,          # Maximum context length
    max_tokens=2048,     # Maximum number of tokens to generate
    
    # Generation parameters
    temperature=0.7,     # Higher = more creative, Lower = more focused
    top_p=0.9,          # Nucleus sampling threshold
    top_k=40,           # Top-k sampling threshold
    repeat_penalty=1.1,  # Penalty for repeating tokens
    
    # Performance parameters
    n_gpu_layers=32,    # Number of layers to offload to GPU
    n_batch=512,        # Batch size for prompt processing
    n_threads=4,        # Number of CPU threads to use
    
    # Streaming and callback configuration
    streaming=True,
    callback_manager=callback_manager,
    verbose=True,       # Enable verbose output
)



prompt = nikki.nikki_prompt_generic


st.set_page_config(page_title="Nikki")
st.title("Nikki")

# Session State
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Hi! I'm Nikki, How are you doing?"),
    ]



def get_response(user_query, chat_history):

    # history is limited to 25 messages
    formatted_history = "\n".join([f"{'Human' if isinstance(msg, HumanMessage) else 'AI'}: {msg.content}" for msg in chat_history[-25:]])  


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
            
    if len(st.session_state.chat_history) > MAX_MESSAGES:
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


