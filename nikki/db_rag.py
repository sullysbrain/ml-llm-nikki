"""
File: db_rag.py
Author: Scott Sullivan
Created: 2024-05-06
Description:
    This module builds the conversational langchain RAG for the chatbot.
    The chain is built using the Ollama LLM, a prompt template, and a RAG retriever.

Functions:
    build_rag(): returns a RAG retriever
"""

import dotenv, file_helper, os
dotenv.load_dotenv()

from sentence_transformers import SentenceTransformer
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings

# from langchain_huggingface import HuggingFaceEmbeddings

from langchain_community.vectorstores import Chroma
from langchain.schema.runnable import RunnablePassthrough

from langchain.prompts import PromptTemplate
from langchain.chains.conversation.prompt import PROMPT
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter



def build_rag(model_name: str, database_directory: str):
    # RAG Retriever
    # load saved chroma vector database
    embedding_function = SentenceTransformerEmbeddings(model_name=model_name)
    reports_vector_db = Chroma(persist_directory=database_directory, embedding_function=embedding_function)
    retriever  = reports_vector_db.as_retriever(k=10)
    return retriever



