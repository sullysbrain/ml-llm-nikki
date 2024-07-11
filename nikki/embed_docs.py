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
from constants import MARKDOWN_REPORTS_PATH, EMBED_MODEL, REPORTS_CHROMA_PATH, REPORTS_PATH
import dotenv, os, file_helper
import glob, json, datetime
dotenv.load_dotenv()
import re

# Vector Store
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
# from langchain_huggingface import HuggingFaceEmbeddings

from langchain_text_splitters import RecursiveCharacterTextSplitter

# from langchain.document_loaders import TextLoader
from langchain_community.document_loaders import TextLoader
from sentence_transformers import SentenceTransformer

from chromadb.utils import embedding_functions
from collections import namedtuple

# Load Docments to Embed
directory_path = "./_private/reports/"

theater_general_pattern = os.path.join(directory_path, "stage_references.md")
report_pattern = os.path.join(directory_path, "report_*.md")
control_pattern = os.path.join(directory_path, "control*")

general_background_files = glob.glob(theater_general_pattern)
report_files = glob.glob(report_pattern)
control_files = glob.glob(control_pattern)

report_files_sorted = sorted(report_files)
file_list = general_background_files + report_files_sorted


text_data_list = []

# Function to read Markdown files and extract text content
def read_markdown_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

Document = namedtuple('Document', ['page_content', 'metadata'])

def extract_date(filename):
    # Regular expression pattern to match the date at the end of the filename
    pattern = r'_(\d{8})\.md$'
    
    # Search for the pattern in the filename
    match = re.search(pattern, filename)
    
    # If a match is found, return the date string
    if match:
        return match.group(1)
    else:
        return None

docs = []
for file in file_list:
    # filepath = os.path.join(directory, filename)
    loader = TextLoader(file)
    doc = loader.load()[0]
    
    # Extract date from filename
    date = extract_date(file)

    # Add metadata
    doc.metadata["source"] = file
    if date != None:
        doc.metadata["date"] = date
    
    docs.append(doc)
    print(f"Loading data from {doc.metadata['source']}")


print(f"Loaded {len(docs)} documents.\n")


# Split into Chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1024, 
    chunk_overlap=64, 
    is_separator_regex=False
)

# report_text_split = text_splitter.split_documents(text_data_list)
report_text_split = text_splitter.split_documents(docs)

docs_to_embed = report_text_split

# Initialize the Sentence Transformer Model for Embeddings
model = SentenceTransformer(EMBED_MODEL)
embedding_function = SentenceTransformerEmbeddings(model_name=EMBED_MODEL)

db2 = Chroma.from_documents(docs_to_embed, embedding_function, persist_directory=REPORTS_CHROMA_PATH)

current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
line_to_append = f"{current_time} - Embedding completed\n"
with open('embeddings_log.txt', 'a') as file:
    file.write(line_to_append)


print("Embedding complete.\n")




# texts = text_splitter.split_text(reports[0].page_content)
# text_docs_reports = text_splitter.split_documents(reports)
# text_docs2_chronicles = text_splitter.split_documents(control_chronicles)
# text_docs2_raynok = text_splitter.split_documents(raynok_report)
