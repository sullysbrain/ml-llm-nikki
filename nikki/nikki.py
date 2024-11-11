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

# Streamlit
import streamlit as st
from constants import REPORTS_CHROMA_PATH, EMBED_MODEL, LANGUAGE_CHROMO_PATH

# Prompts
from langchain.prompts import PromptTemplate
from langchain.chains.conversation.prompt import PROMPT

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




# Local Imports
import db_llm_builder as llm_builder
import rag.db_rag as rag_builder
import rag.prompts.nikki_personality as nikki

# import _private.template_ae as ae_chat


st.set_page_config(page_title="Nikki")
st.title("Nikki")

llm = llm_builder.build_llm(transformer_name="mixtral:8x7b")
prompt = build_prompt(nikki.nikki_prompt_generic)

# calculate_string_tool = CalculateStringTool()
# tool_names=[calculate_string_tool]
# tools = load_tools(tool_names, llm=llm)
# llm.tool_chain = tools


def format_docs(docs):
    return "\n\n".join([d.page_content for d in docs])

def get_response(user_query, chat_history):

    #retriever = rag_builder.build_rag(model_name=EMBED_MODEL, database_directory=REPORTS_CHROMA_PATH)
    # chain = (
    #     ({"context": retriever, "user_question": RunnablePassthrough()})
    #     | prompt
    #     | llm
    #     | StrOutputParser()
    # )
    # return chain.stream(user_query)

    chain = prompt | llm | StrOutputParser()

    return chain.stream(
        {"chat_history": chat_history, "user_question": RunnablePassthrough()}
    )

    

# Session State
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Hi! I'm Nikki. How can I help you?"),
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

    st.session_state.chat_history.append(AIMessage(content=response))





