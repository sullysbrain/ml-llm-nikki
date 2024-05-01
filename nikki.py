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
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema.runnable import RunnablePassthrough, RunnableParallel

# LLM - Ollama
from langchain_community.llms import Ollama
from langchain_experimental.llms.ollama_functions import OllamaFunctions
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# Prompts
from langchain.prompts import PromptTemplate
from langchain.chains.conversation.prompt import PROMPT
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser


# Local Imports
import build_chain

# Build the Chain
# llm = build_chain.build_llm()
# prompt = build_chain.build_prompts()
# retriever = build_chain.build_rag()
# chain = build_chain.build_chain(llm, prompt, retriever)


# Streamlit
import streamlit as st
from constants import REPORTS_CHROMA_PATH, EMBED_MODEL

st.set_page_config(page_title="A&E Chatbot")
st.title("A&E Chatbot")

def get_response(user_query, chat_history):
    prompt_template = """
    You are a helpful assistant. Answer the questions from this context: {context}
    
    Answer the following questions considering the history of the conversation:
    Chat history:
    
    User question: {user_question}
    """

    prompt = ChatPromptTemplate.from_template(prompt_template)

    llm = build_chain.build_llm()

    embedding_function = SentenceTransformerEmbeddings(model_name=EMBED_MODEL)
    reports_vector_db = Chroma(persist_directory=REPORTS_CHROMA_PATH, embedding_function=embedding_function)
    reports_retriever  = reports_vector_db.as_retriever(k=10)

    chain = (
        {"context": reports_retriever, "user_question": RunnablePassthrough()}
        | prompt 
        | llm 
        | StrOutputParser()
    )

    response = chain.stream(user_query)
    
    return response
    

# Session State
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Hello, I am the AE Chatbot. How can I help you?"),
    ]
    
# Conversation
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




# Streamlit
# import streamlit as st
# st.title('The AE AI Assistant')
# with st.form(key='my_form'):
#     text = st.text_area('Chat here:', value='', placeholder='Ask your question', height=200)
#     submitted = st.form_submit_button(label='Submit')
#     if submitted:
#         st.info(chain.invoke(text))


# # Terminal
# while True:
#     user_input = input("\n\n > ")
#     if user_input.lower() == 'quit':
#         break
#     chain.invoke(user_input)





# Chain
# llm = OllamaFunctions(
#     model="llama3:8b", 
#     temperature=1,    
#     stop=["<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>", "<|reserved_special_token"],
#     callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
# )


# History Prompts
# template = """The following is a friendly conversation between a human and an AI named NIKKI. NIKKI is talkative and provides lots of specific details from its context. If NIKKI does not know the answer to a question, it truthfully says it does not know.
# Current conversation:
# {history}
# Human: {input}
# NIKKI:"""



# Memory
#memory = ConversationSummaryBufferMemory(llm=llm, max_token_limit=100)
#memory = ConversationBufferMemory(llm=llm, max_token_limit=100)
# memory=ConversationBufferMemory(ai_prefix="NIKKI")

# conversation = ConversationChain(
#     llm=llm, 
#     memory = memory,
#     verbose=False,
#     prompt=full_prompt,
# )






