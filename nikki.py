from langchain_community.llms import LlamaCpp
from langchain.chains import LLMChain, ConversationalRetrievalChain

from langchain_community.llms import Ollama
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

from langchain.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_community.document_loaders.markdown import UnstructuredMarkdownLoader
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings

import nikki_templates
import file_helper
from constants import MARKDOWN_REPORTS_PATH, ESTOP_LOG_PATH


# llm = LlamaCpp(
#     model_path="models/Meta-Llama-3-8B.Q6_K.gguf",
#     n_gpu_layers=1,
#     n_batch=512,
#     n_ctx=2048,
#     f16_kv=True,
#     callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
#     verbose=False,  # Enable detailed logging for debugging
# )

# Load LLM
# Ollama API options: "llama3:8b", "llama2:13b", "llama3:8b", "mixtral:8x7b", "qwen:32b"
llm = Ollama(
    model="mixtral:8x7b", 
    callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]))

# Load Reference Docs, Tokenize, Vector Embeddings
docs = file_helper.read_markdown_file(MARKDOWN_REPORTS_PATH)
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1024, 
    chunk_overlap=64, 
    is_separator_regex=False)

texts = text_splitter.split_text(docs[0].page_content)
report_docs = ' '.join(texts)

# Load Prompts
user_prompt = ChatPromptTemplate.from_messages([("human", "{question}")])
full_prompt = nikki_templates.nikki_prompt + user_prompt

# Build Chain
chain = full_prompt | llm | StrOutputParser()


while True:
    user_input = input("\n\n > ")
    # user_input = input("\n > ")
    if user_input.lower() == 'quit':
        break

    chain.invoke({"reports": report_docs, "question": user_input})
















