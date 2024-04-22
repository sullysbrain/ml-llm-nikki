from langchain.chains import LLMChain, ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

from langchain_community.llms import Ollama
from langchain_community.chat_models import ChatOllama

from langchain_core.output_parsers import StrOutputParser
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

from langchain_community.embeddings import HuggingFaceEmbeddings

import nikki_templates
import file_helper
from constants import MARKDOWN_REPORTS_PATH, ESTOP_LOG_PATH



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

report_docs = file_helper.read_markdown_file(MARKDOWN_REPORTS_PATH)

while True:
    user_input = input("\n\n > ")
    # user_input = input("\n > ")
    if user_input.lower() == 'quit':
        break

    user_prompt = ChatPromptTemplate.from_messages([("human", "{question}")])
    full_prompt = nikki_templates.nikki_prompt + user_prompt
    chain = full_prompt | llm | StrOutputParser()

    chain.invoke({"reports": report_docs, "question": user_input})








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








