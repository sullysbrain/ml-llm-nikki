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

# Vector Store
from langchain.schema.runnable import RunnablePassthrough, RunnableParallel, RunnableLambda

# Agents and Tools
# from langchain.tools import ToolChain
from langchain.chains import LLMChain

# Prompts
from langchain.prompts import PromptTemplate
from langchain.chains.conversation.prompt import PROMPT
from _private.nikki_private import nikki_friend_private
from rag.prompts.nikki_personality import nikki_tutor_prompt_template

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser

# Local Imports
import models.llm.db_llm_builder as llm_builder
import rag.db_rag as rag_builder
# import rag.prompts.nikki_personality as nikki
# import _private.template_ae as ae_chat

# import rag.agents.db_agent as db_agent

from langchain.tools import BaseTool
# from langchain.tools import ToolChain
# from langchain.agents import load_tools
# from rag.agents.db_agent import CalculateStringTool

# Streamlit
import streamlit as st
from constants import REPORTS_CHROMA_PATH, EMBED_MODEL, LANGUAGE_CHROMO_PATH


st.set_page_config(page_title="Chatbot")
st.title("Chatbot")
llm = llm_builder.build_llm(transformer_name="mixtral:8x7b")


# calculate_string_tool = CalculateStringTool()
# tool_names=[calculate_string_tool]
# tools = load_tools(tool_names, llm=llm)
# llm.tool_chain = tools



def format_docs(docs):
    return "\n\n".join([d.page_content for d in docs])

def get_response(user_query, chat_history):
    # prompt = rag_builder.build_prompt(ae_chat.ae_prompt_template)
    
    prompt = rag_builder.build_prompt(nikki_tutor_prompt_template)
    # prompt = build_chain.build_prompts(nikki.nikki_tutor_prompt_template)

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





