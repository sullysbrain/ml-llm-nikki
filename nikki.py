from langchain.chains import LLMChain, ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
from langchain_community.llms import Ollama
from langchain_community.chat_models import ChatOllama

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
#from langchain_community.callbacks import get_openai_callback
import textwrap
import csv

import nikki_templates


# Read CSV file content
# csv_texts = []
# with open('_private/logs/SS_Estop_log.csv', newline='', encoding='utf-8') as csvfile:
#     reader = csv.reader(csvfile)
#     for row in reader:
#         # Assuming each row is a list of columns and you concatenate them into one string per row
#         csv_texts.append(' '.join(row))

loader = UnstructuredMarkdownLoader("_private/reports_all.md")
docs = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=64, is_separator_regex=False)
texts = text_splitter.split_text(docs[0].page_content)
# Convert texts list into a single string of all items
texts_singletext = ' '.join(texts)


# General
user_question_template = ChatPromptTemplate.from_messages(
    [
        HumanMessagePromptTemplate.from_template("{user_input}")
    ]
)


def chat_with_nikki():
    print("Chat with NIKKI (type 'quit' to end the conversation):")

    # Ollama API  | options: "llama2:13b", "llama3:8b", "mixtral:8x7b", "qwen:32b"
    llm = Ollama(model="llama3:8b", callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]))
    prompt = nikki_templates.nikki_template + user_question_template
    chain = prompt | llm | StrOutputParser()

    print("LLM Loaded\n")

    while True:
        user_input = input("\n\nYou: ")
        if user_input.lower() == 'quit':
            break

        # Ollama API
        #user_context=reports_split
        formatted_chat = prompt.format(user_input=user_input)  

        chain.invoke(formatted_chat)
        

# Example usage
if __name__ == "__main__":
    chat_with_nikki()





    







