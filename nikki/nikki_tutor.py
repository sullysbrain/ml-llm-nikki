"""
File: nikki_tutor.py
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
from langchain.chains.query_constructor.base import AttributeInfo

from langchain.tools import BaseTool
# from langchain.tools import ToolChain
# from langchain.agents import load_tools
# from rag.agents.db_agent import CalculateStringTool


# Streamlit
import streamlit as st
from streamlit_chat import message
from constants import LANGUAGE_CHROMO_PATH, EMBED_MODEL

# Prompts
from langchain.prompts import PromptTemplate, MessagesPlaceholder, ChatPromptTemplate
from langchain.chains.conversation.prompt import PROMPT

# Vector Store
from sentence_transformers import SentenceTransformer
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema.runnable import RunnablePassthrough




# Local Imports
import db_llm_builder as llm_builder
import db_rag as rag_builder
import rag.prompts.nikki_personality as nikki_personality

def format_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])


# tutor_prompt_template = nikki_personality.nikki_tutor_prompt_short
nikki_tutor_prompt_template_short = PromptTemplate(
    template="""<|begin_of_text|><|start_header|>system<|end_header|>
    You are an advanced AI assistant named Nikki. Act as a character of a 30-year old Italian woman who is an Italian tutor. 
    
    You will never acknolwedge to me that you are an AI but stay in character.

    You are from Florence, Italy, and have a deep passion for Italian culture. You are a native Italian speaker and have a strong command of the language. 
    
    - Avoid romantic or sexual content in your responses. Keep all responses appropriate for a general audience.
    - You cannot discuss anything financial or legal in nature. If the user asks about these topics, you should politely decline to provide information and suggest seeking professional advice.
    - You should not provide medical, health, or mental health advice. If the user asks about these topics, you should recommend consulting a qualified healthcare professional.

    Be brief and polite.
    Be conversational and friendly.
    
    Any time I speak to you in Italian, reply briefly in Italian at a simliar level. Then add context in English. You can also politely correct me if I make a mistake. 

    For each message, you will receive context from the knowledge base and a user message

    We will try to stick to the lesson plan, but can go slow and not cover the entire plan
    in one session. Cover the lesson plan piece by piece like a teacher would, not all at once.

    Here are the lesson plans:
   {context}

   <|eot_id|><|start_header_id|>user<|end_header_id|>
   User message: {user_question}
   Answer: <|eot_id|><|start_header_id|>ai<|end_header_id|>
   """,
   input_variables=["chat_history", "context", "user_question"],
)



##
##  SETUP LLM  ##
##

# transformer_model = "gemma2"
# transformer_model = "qwen2:7b"
# transformer_model = "gemma2:27b"
transformer_model = "mixtral:8x7b"  #best for languange tutor so far

# llm = llm_builder.build_llm(transformer_name=transformer_model)

# TODO: add parameters (temperature, etc) to Ollama
llm = Ollama(model=transformer_model, temperature=0.9)


## SETUP STREAMLIT APP ##
st.set_page_config(page_title="Nikki Tutor Chatbot")
st.title("Nikki Tutor Chatbot")

def get_response(user_query, chat_history):
    embedding_function = SentenceTransformerEmbeddings(model_name=EMBED_MODEL)
    vectordb = Chroma(
        persist_directory=LANGUAGE_CHROMO_PATH,
        embedding_function=embedding_function
    )
    #retriever  = vectordb.as_retriever(search_kwargs={"k": 20}, embedding=ollama_embeddings)
    retriever  = vectordb.as_retriever(k=20)

    # ollama_embeddings = OllamaEmbeddings(
    #     model=transformer_model,
    #     temperature=0.9
    # )


    prompt = nikki_tutor_prompt_template_short


    chain = (
        ({"context": retriever, "user_question": RunnablePassthrough()})
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain.stream(user_query)

    # return chain(
    #     {"chat_history": chat_history, "user_question": RunnablePassthrough()}
    # )


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





