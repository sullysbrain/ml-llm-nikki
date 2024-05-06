"""
File: build_chain.py
Author: Scott Sullivan
Created: 2024-05-01
Description:
    This module builds the conversational langchain chain for the chatbot.
    The chain is built using the Ollama LLM, a prompt template, and a RAG retriever.

Functions:
    build_llm(): returns an Ollama LLM
    build_prompts(): returns a prompt template
    build_rag(): returns a RAG retriever
    build_chain(llm, prompt, retriever): returns a langchain chain
"""

# Variable Loaders
from constants import MARKDOWN_REPORTS_PATH, ESTOP_LOG_PATH, REPORTS_CHROMA_PATH, EMBED_MODEL
import dotenv, file_helper, os
dotenv.load_dotenv()

from sentence_transformers import SentenceTransformer
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings

from langchain_community.vectorstores import Chroma
from langchain.schema.runnable import RunnablePassthrough

from langchain.chains import LLMChain, ConversationChain
from langchain.chains.conversation.memory import ConversationBufferMemory, ConversationSummaryMemory
from langchain.memory import ConversationSummaryBufferMemory

from langchain_community.llms import Ollama
from langchain_experimental.llms.ollama_functions import OllamaFunctions
from langchain_core.output_parsers import StrOutputParser
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

from langchain.prompts import PromptTemplate
from langchain.chains.conversation.prompt import PROMPT
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Local Imports
import template_nikki as template_nikki


# Load LLM:  
def build_llm(transformer_name: str = "mixtral:8x7b"):
    # Build LLM with Ollama
    llm = Ollama(
        model="mixtral:8x7b", 
        callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
        stop=["<|start_header_id|>", 
              "<|end_header_id|>", 
              "<|eot_id|>", 
              "<|reserved_special_token"]
    )
    return llm

def build_prompts(prompt_template: PromptTemplate):
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

def build_chain(llm, prompt, retriever):
    # Build Chain

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        # {"question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain


