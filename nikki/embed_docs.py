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
from constants import LANGUAGE_LESSON_01, LANGUAGE_CHROMO_PATH, EMBED_MODEL, REPORTS_CHROMA_PATH, REPORTS_PATH
import dotenv, re, datetime
dotenv.load_dotenv()

import os, glob

# Vector Store
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.document_loaders import TextLoader

from collections import namedtuple
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain.document_loaders import TextLoader
# from chromadb.utils import embedding_functions



# Function to read Markdown files and extract text content
def read_markdown_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def get_file_paths(directory_path, patterns):
    files = []
    for pattern in patterns:
        files.extend(glob.glob(os.path.join(directory_path, pattern)))
    return files


# Load Docments to Embed - AE Reports
# directory_path = REPORTS_PATH
# report_patterns = ['stage_*.md', 'report_*.md', 'control_*.md']
# report_files = get_file_paths(directory_path, report_patterns)

# Tutor
pattern = ['ita_*.md']
report_files = get_file_paths(LANGUAGE_LESSON_01, pattern)



Document = namedtuple('Document', ['page_content', 'metadata'])

def extract_date(filename):
    # Regular expression pattern to match the date at the end of the filename
    pattern = r'_(\d{8})\.md$'
    match = re.search(pattern, filename)
    
    # If a match is found, return the date string
    if match:
        return match.group(1)
    else:
        return None

docs = []
for file in report_files:
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
docs_to_embed = text_splitter.split_documents(docs)

# Initialize the Sentence Transformer Model for Embeddings
model = SentenceTransformer(EMBED_MODEL)
embedding_function = SentenceTransformerEmbeddings(
    model_name=EMBED_MODEL)

embedded_db = Chroma.from_documents(
    docs_to_embed, 
    embedding_function, 
    # persist_directory = REPORTS_CHROMA_PATH)
    persist_directory = LANGUAGE_CHROMO_PATH)


current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
line_to_append = f"{current_time} - Embedding completed\n"
with open('embeddings_log.txt', 'a') as file:
    file.write(line_to_append)

print("Embedding complete.\n")




# texts = text_splitter.split_text(reports[0].page_content)
# text_docs_reports = text_splitter.split_documents(reports)
# text_docs2_chronicles = text_splitter.split_documents(control_chronicles)
# text_docs2_raynok = text_splitter.split_documents(raynok_report)
