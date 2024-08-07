"""
File: nikki_ae.py
Author: Scott Sullivan
Created: 2024-05-01
Description:
    This module is the main entry point for the Chatbot.

Functions:
    get_response(user_query, chat_history): returns a response to the user query
    main(): runs the chatbot
"""

# Variable Loaders
import sys, os, re
import dotenv, file_helper
from datetime import datetime
from dateutil.parser import parse
dotenv.load_dotenv()

# Vector Store
from langchain.schema.runnable import RunnablePassthrough, RunnableParallel, RunnableLambda

# Ollama
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings

# Agents and Tools
# from langchain.tools import ToolChain
from langchain.chains import ConversationalRetrievalChain, LLMChain
from langchain.retrievers import TimeWeightedVectorStoreRetriever
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.question_answering import load_qa_chain
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser

from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document
from langchain.chains.query_constructor.base import AttributeInfo

from langchain.tools import BaseTool
# from langchain.tools import ToolChain
# from langchain.agents import load_tools
# from rag.agents.db_agent import CalculateStringTool


# Streamlit
import streamlit as st
from streamlit_chat import message
from constants import REPORTS_CHROMA_PATH, EMBED_MODEL

# Prompts
from langchain.prompts import PromptTemplate, MessagesPlaceholder, ChatPromptTemplate
from langchain.chains.conversation.prompt import PROMPT

# Vector Store
from sentence_transformers import SentenceTransformer
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema.runnable import RunnablePassthrough
from collections import defaultdict


# Local Imports
import db_llm_builder as llm_builder
import db_rag as rag_builder

def format_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])


##
##  SETUP PROMPT TEMPLATE  ##
##
# template="""<|begin_of_text|><|start_header|>system<|end_header|>
#     Your name is Nikki. You are an advanced AI assistant.

#     Important information:
#     {chat_history}

#     Relevant information:
#     {context}

#     <|eot_id|><|start_header_id|>user<|end_header_id|>
#     User message: {user_question}
#     Answer: <|eot_id|><|start_header_id|>ai<|end_header_id|>
#     """
# input_variables=["chat_history", "context", "user_question"]

ae_prompt_template = PromptTemplate(
    template="""<|begin_of_text|><|start_header|>system<|end_header|>
    Your name is Nikki. You are an advanced AI assistant.
    Any dates can be normal dates or in the format 20240626, often in the context of reports.
    The dates are also listed at the start of each report (similar to 'report of July 10').
    If the user asks about any issues, you can assume they are asking about issues that are documented in the reports.
    Unless otherwise asked, you should assume the user is asking about the most recent reports.

    Embedded Reports are as follows:
    {context}
    
    You can reference the chat history as well: 
    {chat_history}

    <|eot_id|><|start_header_id|>user<|end_header_id|>
    User message: {user_question}
    Answer: <|eot_id|><|start_header_id|>ai<|end_header_id|>
    """,
    input_variables=["context", "chat_history", "user_question"],
)

def format_metadata(docs):
    return "\n".join([str(d.metadata) for d in docs])

def format_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])


    




##
##  SETUP LLM  ##
##

# transformer_model = "gemma2"
transformer_model = "qwen2:7b"
# transformer_model = "gemma2:27b"
# transformer_model = "mixtral:8x7b"

llm = Ollama(model=transformer_model, temperature=1.7)
# llm = Ollama(model=transformer_model)

## SETUP STREAMLIT APP ##
st.set_page_config(page_title="AE Chatbot")
st.title("AE Chatbot")
# st.markdown(page_bg_img, unsafe_allow_html=True)
# try :
#     image_url = "logo-new.png"
#     st.sidebar.image(image_url, caption="", use_column_width=True)
# except :   
#     # image_url = "https://static.vecteezy.com/system/resources/previews/010/794/341/non_2x/purple-artificial-intelligence-technology-circuit-file-free-png.png"
#     st.sidebar.image(image_url, caption="", use_column_width=True)


def get_response(user_query, chat_history):
    embedding_function = SentenceTransformerEmbeddings(model_name=EMBED_MODEL)
    vectordb = Chroma(
        persist_directory=REPORTS_CHROMA_PATH,
        embedding_function=embedding_function
    )
    ollama_embeddings = OllamaEmbeddings(
        model=transformer_model,
        temperature=0.9
    )


    # history is limited to 25 messages
    formatted_history = "\n".join([f"{'Human' if isinstance(msg, HumanMessage) else 'AI'}: {msg.content}" for msg in chat_history[-25:]])  

    prompt = ae_prompt_template

    # Extract date from the user query, if exists
    # query_date = extract_date_from_query(user_query)
    # print(f"formatted dates: {query_date}")

    # retriever = vectordb.as_retriever(search_kwargs={'filter': {'date':'20240607'}})
    retriever = vectordb.as_retriever(search_kwargs={"k": 20}, embedding=ollama_embeddings, return_source_documents=True)


    chain = (
        {
            "context": retriever, 
            # "dates": dates,
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

    # Debugging: Print the input data to ensure it's correct
    print(f"Input Data: {input_data}")

    # response = chain.stream(input_data)
    response = chain.stream(user_query)
    return response


# Session State
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Hi! I'm Nikki. What can I help you with?"),
    ]
    
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

        # response = st.write_stream(get_response(user_query, st.session_state.chat_history))
        # response = st.write(get_response(user_query, st.session_state.chat_history))
        # response = get_response(user_query, st.session_state.chat_history)
 
        # st.write(response)
    st.session_state.chat_history.append(AIMessage(content=response))



