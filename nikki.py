import dotenv, file_helper, os
import build_chain

# Variable Loaders
dotenv.load_dotenv()

from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings

from langchain_community.vectorstores import Chroma
from langchain.schema.runnable import RunnablePassthrough

from langchain_community.llms import Ollama
from langchain_experimental.llms.ollama_functions import OllamaFunctions
from langchain_core.output_parsers import StrOutputParser
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

from langchain.prompts import PromptTemplate
from langchain.chains.conversation.prompt import PROMPT
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate



# Streamlit
# import streamlit as st
# st.title('The AE AI Assistant')
# with st.form(key='my_form'):
#     text = st.text_area('Chat here:', 'Ask your question', height=200)
#     submitted = st.form_submit_button(label='Submit')
#     if submitted:
#         st.info(report_chain.invoke(text))


# # Terminal
llm = build_chain.build_llm()
prompt = build_chain.build_prompts()
retriever = build_chain.build_rag()
chain = build_chain.build_chain(llm, prompt, retriever)


while True:
    user_input = input("\n\n > ")
    if user_input.lower() == 'quit':
        break
    chain.invoke(user_input)





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






