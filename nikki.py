from langchain.chains import LLMChain, ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, MessagesPlaceholder
from langchain_community.llms import Ollama
from langchain_core.output_parsers import StrOutputParser

from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_community.document_loaders.markdown import UnstructuredMarkdownLoader
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings

## HISTORY MEMORY
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferMemory
#from langchain.callbacks import get_openai_callback
from langchain_community.callbacks import get_openai_callback
import textwrap

import nikki_templates


# OpenAI specific imports
# from langchain.agents import load_tools, initialize_agent, AgentType
# from langchain_openai import OpenAI
# import os
# import sys
# sys.path.append('_private/')
# from _private import api
# os.environ['OPENAI_API_KEY'] = api.API_KEY




loader = UnstructuredMarkdownLoader("_private/reports_all.md")
docs = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=64, is_separator_regex=False)
texts = text_splitter.split_text(docs[0].page_content)
# Convert texts list into a single string of all items
texts_singletext = ' '.join(texts)

# General
user_question_template = ChatPromptTemplate.from_messages(
    [
        #('system', 'Here is some context for the conversation: {user_context}'),
        ("system", " You will obtain information from the prompts as your primary source of data: {user_context} to answer the user's input:"),
        ("user", "{user_input}")
    ]
)


# OpenAI
# system_message_template = SystemMessagePromptTemplate.from_template(
#     "You are a helpful AI bot. Here is context for our conversation {user_context}."
# )
# # Format the system message
# system_message = system_message_template.format(user_context=texts_singletext)
# chat_template = ChatPromptTemplate.from_messages([
#     system_message_template,
#     # Add other messages here...
# ])
# # Format all the messages in the template
# messages = chat_template.format_messages(user_context=texts_singletext, user_input="How are the batteries?")



complete_context = nikki_templates.nikki_template + user_question_template


def chat_with_nikki():
    print("Chat with NIKKI (type 'quit' to end the conversation):")

    # Ollama API
    # nikki = Ollama(model="llama2:13b", callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]))
    # nikki = Ollama(model="qwen:32b", callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]))
    # nikki = Ollama(model="mixtral:8x22b", callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]))
    nikki = Ollama(model="mixtral:8x7b", callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]))

    # conversation = ConversationChain(llm=nikki)
    # conversation_buf = ConversationChain(
    #     llm=nikki,
    #     memory=ConversationBufferMemory()
    # )

    # OpenAI API
    # llm = OpenAI(openai_api_key=os.getenv("OPENAI_API_KEY"))
    # nikki = LLMChain(prompt=complete_context, llm=llm)    

    while True:
        user_input = input("\n\nYou: ")
        if user_input.lower() == 'quit':
            break

        # Ollama API
        formatted_chat = complete_context.format_messages(user_context=texts_singletext, user_input=user_input)
        print(nikki.invoke(formatted_chat))
        #print(conversation.prompt.template)

        # OpenAI API        
        # qa = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever, return_source_documents=True, combine_docs_chain_kwargs={"prompt": chat_template})
        # print(nikki.invoke(qa))
        

# Example usage
if __name__ == "__main__":
    chat_with_nikki()





    







