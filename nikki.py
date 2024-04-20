from langchain.chains import LLMChain, ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain_core.prompts import FewShotPromptTemplate, ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
from langchain_community.llms import Ollama
from langchain_community.chat_models import ChatOllama

from langchain_community.llms import LlamaCpp

from langchain_core.output_parsers import StrOutputParser

from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_community.document_loaders.markdown import UnstructuredMarkdownLoader
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferMemory, ConversationBufferWindowMemory

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

# Prompt
nikki_prompt = nikki_templates.nikki_prompt


# Load the LlamaCpp language model, adjust GPU usage based on your hardware
# llm = LlamaCpp(
#     # model_path="models/Meta-Llama-3-8B.Q3_K_L.gguf",
#     model_path="models/Meta-Llama-3-8B.Q6_K.gguf",
#     n_gpu_layers=1,
#     n_batch=512,
#     n_ctx=2048,
#     f16_kv=True,
#     callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
#     verbose=False,  # Enable detailed logging for debugging
# )

llm = Ollama(model="mixtral:8x7b", callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]))



# print("Chatbot initialized, ready to chat...\n\n")
while True:
    user_input = input("\n\n > ")
    # user_input = input("\n > ")
    if user_input.lower() == 'quit':
        break


    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You're a expert data scientist and machine learning engineer who offer its expertise and responds to the point and accurately.",
            ),
            ("human", "{question}"),
        ]
    )
    runnable = prompt | llm | StrOutputParser()


    runnable.invoke({"question": user_input})



# Ollama API  | options: "llama2:13b", "llama3:8b", "mixtral:8x7b", "qwen:32b"
# llm = ChatOllama(model="llama3:8b", callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]))

# print(llm.invoke("Tell me a joke"))



# print("Hello, I'm NIKKI, an AI assistant for stage productions questions. Type 'quit' to exit.")
# while True:
#     user_input = input("\n\nYou: ")
#     if user_input.lower() == 'quit':
#         break

#     # Ollama API
#     chain = nikki_prompt | llm
#     chain.invoke({"reports": report_docs, "user_input": user_input})








