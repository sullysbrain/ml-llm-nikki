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
import sys, os, types
import dotenv
dotenv.load_dotenv()


# Ollama
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings

# Langchain
from langchain.schema.runnable import RunnablePassthrough, RunnableParallel, RunnableLambda
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
# from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import Chroma

# Prompts
from langchain.prompts import PromptTemplate, MessagesPlaceholder, ChatPromptTemplate
from langchain.chains.conversation.prompt import PROMPT

# Agents and Tools
# from langchain.retrievers import TimeWeightedVectorStoreRetriever
# from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.question_answering import load_qa_chain
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
# from langchain.chains.query_constructor.base import AttributeInfo

from langchain.tools import BaseTool


# Streamlit
import streamlit as st
from streamlit_chat import message




# Local Imports
from rag.prompts.nikki_personality import nikki_prompt_template_tutor
from constants import LANGUAGE_CHROMO_PATH, EMBED_MODEL


# Helper functions
def format_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])

def format_metadata(docs):
    return "\n".join([str(d.metadata) for d in docs])

def format_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])



##
##  SETUP LLM  ##
##

# transformer_model = "gemma2"
# transformer_model = "gemma2:27b"
# transformer_model = "mixtral:8x7b"  #best for languange tutor so far

# transformer_model = "qwen2:7b"
# transformer_model = "llama3.1:70b"
transformer_model = "llama3.1"

llm = Ollama(model=transformer_model, temperature=0.5)

prompt = nikki_prompt_template_tutor


## SETUP STREAMLIT APP ##
st.set_page_config(page_title="Italian Tutor Chatbot")
st.title("Italian Tutor Chatbot")

# Session State
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Hi! I'm Nikki, your Italian language tutor. What can I help you with?"),
    ]
# track lesson number
# Initialize session state for lesson number if not already set
if 'lesson_number' not in st.session_state:
    st.session_state.lesson_number = None
# Function to update lesson number
def set_lesson_number(lesson_number):
    st.session_state.lesson_number = lesson_number
def extract_lesson_number_instruction(response):
    # This function checks the response for instructions to set the lesson number
    # For example, you can use regular expressions to find "Set lesson number to X"
    import re
    match = re.search(r"start lesson (\d+)", response)
    if match:
        return int(match.group(1))
    return None

# Example input for setting lesson number
# lesson_number = st.number_input('Enter lesson number:', min_value=1, step=1)
# if st.button('Set Lesson Number'):
#     set_lesson_number(lesson_number)


def get_response(user_query, chat_history):
    embedding_function = SentenceTransformerEmbeddings(model_name=EMBED_MODEL)
    vectordb = Chroma(
        persist_directory=LANGUAGE_CHROMO_PATH,
        embedding_function=embedding_function
    )
    ollama_embeddings = OllamaEmbeddings(
        model=transformer_model,
        temperature=0.9
    )

    # history is limited to 25 messages
    formatted_history = "\n".join([f"{'Human' if isinstance(msg, HumanMessage) else 'AI'}: {msg.content}" for msg in chat_history[-25:]])  

    retriever = vectordb.as_retriever(search_kwargs={"k": 10}, embedding=ollama_embeddings, return_source_documents=True)

    # Get relevant docs
    retrieved_docs = retriever.get_relevant_documents(user_query)

    # Filter documents based on the lesson number from session state
    lesson_number = st.session_state.get('lesson_number')
    if lesson_number is not None:
        retrieved_docs = [doc for doc in retrieved_docs if doc.metadata.get('lesson_id') == str(lesson_number)]

    print(f"\n\nLesson Number: {lesson_number}\n\n")        

    # Format context with metadata
    context = ""
    for idx, doc in enumerate(retrieved_docs):
        metadata = doc.metadata
        top_level_header = metadata.get('top_level_header', 'N/A')
        lesson_id = metadata.get('lesson_id', 'N/A')
        language = metadata.get('language', 'N/A')
        level = metadata.get('level', 'N/A')
        content_preview = doc.page_content[:50]
        
        context += (f"Document {idx}:\n"
                    f"Top-level Header: {top_level_header}\n"
                    f"Lesson ID: {lesson_id}\n"
                    f"Language: {language}\n"
                    f"Level: {level}\n"
                    f"Content preview: {content_preview}...\n"
                    "---\n")

    chain = (
        {
            "context": lambda _: context, 
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

    response = chain.stream(input_data)

    # Process the response as a string
    # response = chain.run(input_data)
    # if isinstance(response, types.GeneratorType):
    #     response = ''.join(list(response))

    # Process LLM response to update lesson number if necessary
    # lesson_number_instruction = extract_lesson_number_instruction(response)
    # if lesson_number_instruction:
    #     set_lesson_number(lesson_number_instruction)

    return response



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





