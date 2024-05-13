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

# Prompts
from langchain.prompts import PromptTemplate
from langchain.chains.conversation.prompt import PROMPT

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser

# Local Imports
import models.llm.db_llm_builder as llm_builder
import rag.db_rag as rag_builder
import rag.prompts.nikki_personality as nikki
import _private.template_ae as ae_chat

import rag.agents.db_agent as db_agent

# Streamlit
import streamlit as st
from constants import REPORTS_CHROMA_PATH, EMBED_MODEL, LANGUAGE_CHROMO_PATH


st.set_page_config(page_title="Chatbot")
st.title("Chatbot")
llm = llm_builder.build_llm(transformer_name="mixtral:8x7b")


def format_docs(docs):
    return "\n\n".join([d.page_content for d in docs])

def get_response(user_query, chat_history):
    prompt = rag_builder.build_prompt(ae_chat.ae_prompt_template)
    # prompt = build_chain.build_prompts(nikki.nikki_tutor_prompt_template)

    retriever = rag_builder.build_rag(model_name=EMBED_MODEL, database_directory=REPORTS_CHROMA_PATH)


    # ERROR: 
    # TypeError: Expected a Runnable, callable or dict. Instead got an unsupported type: <class 'list'>
    # TODO: Add agents for SQL access
    # TODO: Get docker / sql running
    
    # create the agent
    tools = [db_agent.prices_retrieval_tool()]
    agent = initialize_agent(llm=llm,
                            tools=tools,
                            agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
                            verbose=True)
    prompt = PromptTemplate(input_variables=['input'], template=template)
    agent.run(prompt.format(input=STORY))


    chain = (
        ({"context": retriever, "user_question": RunnablePassthrough()})
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain.stream(user_query)

    # return chain.stream({
    #     "chat_history": chat_history,
    #     "user_question": user_query,
    # })

    

# Session State
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Hi! How can I help you?"),
    ]
    
# # Conversation
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





