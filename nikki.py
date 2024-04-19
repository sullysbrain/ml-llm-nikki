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
report_docs = ' '.join(texts)

# prompt
nikki_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are an advanced AI assistant. You are a based on the JARVIS AI assistant from the Iron Man movie. You are designed to be intelligent, efficient, and subtly witty. Respond to human queries with concise answers, and a bit of sarcasm and wit. Your name is NIKKI. Your main reference doc is from the reports: {reports}"),
        ("human", "{user_input}")
    ]
)


# Ollama API  | options: "llama2:13b", "llama3:8b", "mixtral:8x7b", "qwen:32b"
llm = ChatOllama(model="qwen:32b", callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]))


while True:
    user_input = input("\n\nYou: ")
    if user_input.lower() == 'quit':
        break

    # Ollama API
    chain = nikki_prompt | llm | StrOutputParser()
    chain.invoke({"reports": report_docs, "user_input": user_input})


#chain = nikki_prompt | llm | StrOutputParser()
#for chunk in chain.stream({"question": user_input}):
#    print(chunk, end="", flush=True)









