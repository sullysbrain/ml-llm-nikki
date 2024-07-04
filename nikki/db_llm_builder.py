"""
File: db_llm_builder.py
Author: Scott Sullivan
Created: 2024-05-01
Description:
    This module builds the conversational langchain chain for the chatbot.
    The chain is built using the Ollama LLM, a prompt template, and a RAG retriever.

Functions:
    build_llm(): returns an Ollama LLM
"""

# Variable Loaders
from constants import MARKDOWN_REPORTS_PATH, ESTOP_LOG_PATH, REPORTS_CHROMA_PATH, EMBED_MODEL
import dotenv, file_helper, os
dotenv.load_dotenv()

from langchain_community.llms import Ollama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler


def build_llm(transformer_name):
    # Build LLM with Ollama
    llm = Ollama(
        model=transformer_name, 
        callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
        stop=["<|start_header_id|>", 
              "<|end_header_id|>", 
              "<|eot_id|>", 
              "<|reserved_special_token"]
    )
    return llm



