import dotenv
import nikki_templates
import file_helper

# Variable Loaders
from constants import MARKDOWN_REPORTS_PATH, ESTOP_LOG_PATH, REPORTS_CHROMA_PATH
from _private.api import API_KEY_OPENAI
dotenv.load_dotenv()

from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.schema.runnable import RunnablePassthrough

from langchain_community.llms import LlamaCpp
from langchain.chains import LLMChain, ConversationChain
from langchain.chains.conversation.memory import ConversationBufferMemory, ConversationSummaryMemory
from langchain.chains.conversation.prompt import PROMPT
from langchain.memory import ConversationSummaryBufferMemory
# Now we can override it and set it to "AI Assistant"

from langchain_community.llms import Ollama
from langchain_core.output_parsers import StrOutputParser
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

from langchain.prompts import PromptTemplate
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_community.document_loaders.markdown import UnstructuredMarkdownLoader
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings



# llm = LlamaCpp(
#     model_path="models/Meta-Llama-3-8B.Q6_K.gguf",
#     n_gpu_layers=1,
#     n_batch=512,
#     n_ctx=2048,
#     f16_kv=True,
#     callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
#     verbose=False,  # Enable detailed logging for debugging
# )

# Load Reference Docs, Tokenize, Vector Embeddings
docs = file_helper.read_markdown_file(MARKDOWN_REPORTS_PATH)
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1024, 
    chunk_overlap=64, 
    is_separator_regex=False)

texts = text_splitter.split_text(docs[0].page_content)
report_docs = ' '.join(texts)
 
# reports_vector_db = Chroma.from_texts(
#     texts, OpenAIEmbeddings(api_key=API_KEY_OPENAI), persist_directory=REPORTS_CHROMA_PATH
# )

reports_vector_db = Chroma(
    persist_directory=REPORTS_CHROMA_PATH,
    embedding_function=OpenAIEmbeddings(api_key=API_KEY_OPENAI),
)


# Load LLM
# Ollama API options: "llama3:8b", "llama2:13b", "llama3:8b", "mixtral:8x7b", "qwen:32b"
llm = Ollama(
    model="llama3:8b", 
    callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
    stop=["<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>", "<|reserved_special_token"])

# Load Prompts
# With History
# template = """The following is a friendly conversation between a human and an AI named NIKKI. NIKKI is talkative and provides lots of specific details from its context. If NIKKI does not know the answer to a question, it truthfully says it does not know.
# Current conversation:
# {history}
# Human: {input}
# NIKKI:"""


review_template_str = """You are NIKKI, an AI assistant whos job is to use report logs to answer questions about a stage show technical log. Use the following context to answer questions. Be as detailed as possible, but don't make up any information that's not from the context. If you don't know an answer, say you don't know.

{context}
"""

review_system_prompt = SystemMessagePromptTemplate(
    prompt=PromptTemplate(
        input_variables=["context"],
        template=review_template_str,
    )
)
review_human_prompt = HumanMessagePromptTemplate(
    prompt=PromptTemplate(
        input_variables=["question"],
        template="{question}",
    )
)

messages = [review_system_prompt, review_human_prompt]

report_prompt_template = ChatPromptTemplate(
    input_variables=["context", "question"],
    messages=messages,
)
reports_retriever  = reports_vector_db.as_retriever(k=10)


# PROMPT = PromptTemplate(input_variables=["history", "input"], template=template)
# prompt_history_template = """\nCurrent conversation:\n{history}\nHuman: {input}\nNIKKI:"""
# prompt_history = ChatPromptTemplate.from_template(prompt_history_template)

# full_prompt = nikki_templates.nikki_prompt + report_docs + prompt_history

output_parser = StrOutputParser()

# Build Chain
# chain = full_prompt | llm | StrOutputParser()
# review_chain = review_prompt_template | llm | output_parser
report_chain = (
    {"context": reports_retriever, "question": RunnablePassthrough()}
    | report_prompt_template
    | llm
    | StrOutputParser()
)


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

while True:
    user_input = input("\n\n > ")
    if user_input.lower() == 'quit':
        break
    
    report_chain.invoke(user_input)
    # review_chain.invoke({"context": report_docs, "question": user_input})


    # conversation.predict(input=user_input)
    # memory.save_context({"input": user_input}, {"output": ai_answer})
















