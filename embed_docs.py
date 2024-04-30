import dotenv, os, file_helper
import nikki_templates

# Variable Loaders
from constants import MARKDOWN_REPORTS_PATH, ESTOP_LOG_PATH, REPORTS_CHROMA_PATH, EMBED_MODEL
dotenv.load_dotenv()

from langchain_community.vectorstores import Chroma
from sentence_transformers import SentenceTransformer
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings

# from chromadb.utils import embedding_functions
# from chromadb.config import Settings
# from chromadb import Client

# from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_community.embeddings import HuggingFaceEmbeddings


docs = file_helper.read_markdown_file(MARKDOWN_REPORTS_PATH)
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1024, 
    chunk_overlap=64, 
    is_separator_regex=False)

texts = text_splitter.split_text(docs[0].page_content)
text_docs = text_splitter.split_documents(docs)
report_docs = ' '.join(texts)


# Initialize the Sentence Transformer Model for Embeddings
# model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
model = SentenceTransformer(EMBED_MODEL)

# create the open-source embedding function
embedding_function = SentenceTransformerEmbeddings(model_name=EMBED_MODEL)

# save to disk
db2 = Chroma.from_documents(text_docs, embedding_function, persist_directory=REPORTS_CHROMA_PATH)

print("Embedding complete.\n")


