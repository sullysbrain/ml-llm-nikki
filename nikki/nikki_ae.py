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
from langchain.chains import ConversationalRetrievalChain, LLMChain
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser

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



# Local Imports
import db_llm_builder as llm_builder
import db_rag as rag_builder

def format_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])



template="""<|begin_of_text|><|start_header|>system<|end_header|>
    Your name is Nikki. You are an advanced AI assistant.

    Important information:
    {chat_history}

    Relevant information:
    {context}

    <|eot_id|><|start_header_id|>user<|end_header_id|>
    User message: {user_question}
    Answer: <|eot_id|><|start_header_id|>ai<|end_header_id|>
    """
input_variables=["chat_history", "context", "user_question"]
ae_prompt_template = PromptTemplate(
    template=template,
    input_variables=input_variables
)



st.set_page_config(page_title="AE Chatbot")
st.title("AE Chatbot")

# transformer_model = "mixtral:8x7b"
# transformer_model = "llama3:8b"
transformer_model = "gemma2"

llm = llm_builder.build_llm(transformer_name=transformer_model)
embedding_function = SentenceTransformerEmbeddings(model_name=EMBED_MODEL)
vectordb = Chroma(
    persist_directory=REPORTS_CHROMA_PATH,
    embedding_function=embedding_function
)
retriever  = vectordb.as_retriever(k=10)


# Contextualize question
contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, just "
    "reformulate it if needed and otherwise return it as is."
)
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)

# Answer question
qa_system_prompt = (
    "You are an assistant for question-answering tasks. Use "
    "the following pieces of retrieved context to answer the "
    "question. If you don't know the answer, just say that you "
    "don't know. Use three sentences maximum and keep the answer "
    "concise."
    "
)




# def get_response(user_question, chat_history):

#     embedding_function = SentenceTransformerEmbeddings(model_name=EMBED_MODEL)
#     vectordb = Chroma(
#         persist_directory=REPORTS_CHROMA_PATH,
#         embedding_function=embedding_function
#     )
#     retriever  = vectordb.as_retriever(k=10)
    # history = "Your favorite color is blue."
    # input_param = {"history": history, "context": retriever, "user_question": RunnablePassthrough()}

    # chain = (
    #     {
    #         "history": chat_history, 
    #         "context": retriever, 
    #         "user_question": RunnablePassthrough()
    #     }
    #     | ae_prompt_template
    #     | llm
    #     | StrOutputParser()
    # )
    # return chain.stream(user_question)


# chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever)



# Function for conversational chat
def conversational_chat(query):
    result = chain({"question": query, "chat_history": st.session_state['history']})
    st.session_state['history'].append((query, result["answer"]))
    return result["answer"]

    
# Initialize chat history
if 'history' not in st.session_state:
    st.session_state['history'] = []

# Initialize messages
if 'generated' not in st.session_state:
    st.session_state['generated'] = ["Hello ! Ask me about the show reports!"]

if 'past' not in st.session_state:
    st.session_state['past'] = ["Hey!"]




# Session State
# if "chat_history" not in st.session_state:
#     st.session_state.chat_history = [
#         AIMessage(content="Hi! I'm Nikki. What can I help you with?"),
#     ]
    
# # # # Conversation
# for message in st.session_state.chat_history:
#     if isinstance(message, AIMessage):
#         with st.chat_message("AI"):
#             st.write(message.content)
#     elif isinstance(message, HumanMessage):
#         with st.chat_message("Human"):
#             st.write(message.content)

# Create containers for chat history and user input
response_container = st.container()
container = st.container()

# User input form
with container:
    with st.form(key='my_form', clear_on_submit=True):
        user_input = st.text_input("Query:", placeholder="Talk to your data 👉 (:", key='input')
        submit_button = st.form_submit_button(label='Send')

    if submit_button and user_input:
        output = conversational_chat(user_input)
        st.session_state['past'].append(user_input)
        st.session_state['generated'].append(output)

# Display chat history
if st.session_state['generated']:
    with response_container:
        for i in range(len(st.session_state['generated'])):
            message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="adventurer-neutral")
            message(st.session_state["generated"][i], key=str(i), avatar_style="bottts")




# User Input
# user_query = st.chat_input("Type your message here...")
# if user_query is not None and user_query != "":
#     st.session_state.chat_history.append(HumanMessage(content=user_query))

#     with st.chat_message("Human"):
#         st.markdown(user_query)

#     with st.chat_message("AI"):
#         response = st.write_stream(get_response(user_query, st.session_state.chat_history))
#         # response = get_response(user_query)
 
#         # st.write(response)
#     st.session_state.chat_history.append(AIMessage(content=response))

    # print("\n\nNEW Chat History **** \n\n", st.session_state.chat_history)


