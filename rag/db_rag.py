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

from langchain_community.vectorstores import Chroma
from langchain.schema.runnable import RunnablePassthrough

from langchain.prompts import PromptTemplate
from langchain.chains.conversation.prompt import PROMPT
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter


def build_prompt(prompt_template: PromptTemplate):
    # Prompt Templates
    # system_prompt = SystemMessagePromptTemplate(
    #     prompt=PromptTemplate(
    #         input_variables=["context"],
    #         template=template_nikki.nikki_prompt_str,
    #     )
    # )
    # human_prompt = HumanMessagePromptTemplate(
    #     prompt=PromptTemplate(
    #         input_variables=["question"],
    #         template="{question}",
    #     )
    # )
    prompt = prompt_template
    # messages = [system_prompt]
    # prompt = ChatPromptTemplate(
    #     input_variables=["context", "user_question"],
    #     messages=messages,
    # )
    return prompt


def build_rag(model_name: str, database_directory: str):
    # RAG Retriever
    # load saved chroma vector database
    embedding_function = SentenceTransformerEmbeddings(model_name=model_name)
    reports_vector_db = Chroma(persist_directory=database_directory, embedding_function=embedding_function)
    retriever  = reports_vector_db.as_retriever(k=10)
    return retriever



