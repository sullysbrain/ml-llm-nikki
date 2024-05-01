"""
File: embed_docs.py
Author: Scott Sullivan
Created: 2024-05-01
Description:
    This module reads in markdown files and embeds the text using a Sentence Transformer model.
    The resulting vector database is saved as a ChromaDB and can be used as a retriever for a conversational AI.    

Functions:
    no defined functions
"""

# Variable Loaders
from constants import MARKDOWN_REPORTS_PATH, ESTOP_LOG_PATH, REPORTS_CHROMA_PATH, MARKDOWN_CONTROL_PATH, EMBED_MODEL
import dotenv, os, file_helper
dotenv.load_dotenv()

# Vector Store
from langchain_community.vectorstores import Chroma
from sentence_transformers import SentenceTransformer
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Local Imports
import nikki_templates

# Load the documents
reports = file_helper.read_markdown_file(MARKDOWN_REPORTS_PATH)
control_chronicles = file_helper.read_markdown_file(MARKDOWN_CONTROL_PATH)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1024, 
    chunk_overlap=64, 
    is_separator_regex=False
)

# texts = text_splitter.split_text(reports[0].page_content)
text_docs_reports = text_splitter.split_documents(reports)
text_docs2_chronicles = text_splitter.split_documents(control_chronicles)
text_docs_all = text_docs_reports + text_docs2_chronicles

# Initialize the Sentence Transformer Model for Embeddings
model = SentenceTransformer(EMBED_MODEL)
embedding_function = SentenceTransformerEmbeddings(model_name=EMBED_MODEL)
db2 = Chroma.from_documents(text_docs_all, embedding_function, persist_directory=REPORTS_CHROMA_PATH)

print("Embedding complete.\n")


