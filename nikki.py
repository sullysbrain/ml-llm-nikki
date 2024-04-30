import dotenv, file_helper, os
import nikki_templates

# Variable Loaders
from constants import MARKDOWN_REPORTS_PATH, ESTOP_LOG_PATH, REPORTS_CHROMA_PATH, EMBED_MODEL
dotenv.load_dotenv()

from sentence_transformers import SentenceTransformer
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings

from langchain_community.vectorstores import Chroma
from langchain.schema.runnable import RunnablePassthrough

from langchain.chains import LLMChain, ConversationChain
from langchain.chains.conversation.memory import ConversationBufferMemory, ConversationSummaryMemory
from langchain.memory import ConversationSummaryBufferMemory

from langchain_community.llms import Ollama
from langchain_experimental.llms.ollama_functions import OllamaFunctions
from langchain_core.output_parsers import StrOutputParser
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

from langchain.prompts import PromptTemplate
from langchain.chains.conversation.prompt import PROMPT
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter

 
# Load LLM:  
# OPTIONS:: "llama3:8b", "llama2:13b", "llama3:8b", "mixtral:8x7b", "qwen:32b"

llm = Ollama(
    model="llama3:8b", 
    callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
    stop=["<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>", "<|reserved_special_token"]
)


# Prompt Templates
review_system_prompt = SystemMessagePromptTemplate(
    prompt=PromptTemplate(
        input_variables=["context"],
        template=nikki_templates.nikki_prompt_str,
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

# RAG Retriever
# load saved chroma vector database
embedding_function = SentenceTransformerEmbeddings(model_name=EMBED_MODEL)
reports_vector_db = Chroma(persist_directory=REPORTS_CHROMA_PATH, embedding_function=embedding_function)
reports_retriever  = reports_vector_db.as_retriever(k=10)


# Build Chain
report_chain = (
    {"context": reports_retriever, "question": RunnablePassthrough()}
    | report_prompt_template
    | llm
    | StrOutputParser()
)


while True:
    user_input = input("\n\n > ")
    if user_input.lower() == 'quit':
        break
    
    report_chain.invoke(user_input)

    # review_chain.invoke({"context": report_docs, "question": user_input})

    # conversation.predict(input=user_input)
    # memory.save_context({"input": user_input}, {"output": ai_answer})








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





