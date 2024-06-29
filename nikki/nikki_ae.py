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
import sys, os
import dotenv, file_helper
dotenv.load_dotenv()

# Vector Store
from langchain.schema.runnable import RunnablePassthrough, RunnableParallel, RunnableLambda

# Agents and Tools
# from langchain.tools import ToolChain
from langchain.chains import LLMChain
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser

from langchain.tools import BaseTool
# from langchain.tools import ToolChain
# from langchain.agents import load_tools
# from rag.agents.db_agent import CalculateStringTool

# Streamlit
import streamlit as st
from constants import REPORTS_CHROMA_PATH, EMBED_MODEL

# Prompts
from langchain.prompts import PromptTemplate
from langchain.chains.conversation.prompt import PROMPT

# Vector Store
from sentence_transformers import SentenceTransformer
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema.runnable import RunnablePassthrough



# Local Imports
import db_llm_builder as llm_builder
import db_rag as rag_builder


ae_prompt_template = PromptTemplate(
    template="""<|begin_of_text|><|start_header|>system<|end_header|>
    Your name is Nikki. You are an advanced AI assistant.

    Context:

    Relevant information:
    {context}

    <|eot_id|><|start_header_id|>user<|end_header_id|>
    User message: {user_question}
    Answer: <|eot_id|><|start_header_id|>ai<|end_header_id|>
    """,
    input_variables=["chat_history", "context", "user_question"],
)




st.set_page_config(page_title="AE Chatbot")
st.title("AE Chatbot")

# transformer_model = "mixtral:8x7b"
transformer_model = "llama3:8b"

llm = llm_builder.build_llm(transformer_name=transformer_model)
prompt = ae_prompt_template

def format_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])


def get_response(user_question, chat_history):

    embedding_function = SentenceTransformerEmbeddings(model_name=EMBED_MODEL)
    vectordb = Chroma(
        persist_directory=REPORTS_CHROMA_PATH,
        embedding_function=embedding_function
    )
    retriever  = vectordb.as_retriever(k=10)

    chain = (
        ({"context": retriever, "user_question": RunnablePassthrough()})
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain.stream(user_question)

    

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
        # response = get_response(user_query)
 
        # st.write(response)
    st.session_state.chat_history.append(AIMessage(content=response))

    # print("\n\nNEW Chat History **** \n\n", st.session_state.chat_history)


