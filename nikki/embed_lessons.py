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
from constants import LANGUAGE_LESSON_PATH, LANGUAGE_CHROMO_PATH, EMBED_MODEL
import dotenv, re, datetime
dotenv.load_dotenv()

import os, glob, argparse, yaml, json

# Vector Store
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownTextSplitter

from itertools import islice
from langchain_community.document_loaders import TextLoader
from typing import List, Dict, Union

from langchain.schema import Document

from collections import namedtuple


# Constants
directory_path = LANGUAGE_LESSON_PATH


def get_file_paths(directory_path, patterns):
    files = []
    for pattern in patterns:
        files.extend(glob.glob(os.path.join(directory_path, pattern)))
    return files


def read_markdown_files(directory):
    markdown_docs = []
    for filename in glob.glob(os.path.join(directory, 'ita_*.md')):
        with open(filename, 'r', encoding='utf-8') as file:
            content = file.read()
            markdown_docs.append(content)
    return markdown_docs


# Read the markdown files
markdown_docs = read_markdown_files(directory_path)

# Print the number of documents read
print(f"Read {len(markdown_docs)} markdown documents.")



# Split into Chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1024, 
    chunk_overlap=64, 
    is_separator_regex=False
)
Document = namedtuple('Document', ['page_content', 'metadata'])


markdown_splitter = MarkdownTextSplitter(chunk_size=500, chunk_overlap=50)


# Process each markdown document
all_chunks = []
for idx, markdown_content in enumerate(markdown_docs, start=1):
    # Split the markdown content
    chunks = markdown_splitter.split_text(markdown_content)
    
    # Create Document objects with metadata
    doc_chunks = [
        Document(
            page_content=chunk,
            metadata={
                "lesson_id": idx,
                "source": f"Lesson {idx}"
            }
        ) for chunk in chunks
    ]
    
    all_chunks.extend(doc_chunks)




# for d in all_docs:
#     print(f"\n\nDoc:\n{d}\n\n")

# Initialize the Sentence Transformer Model for Embeddings
model = SentenceTransformer(EMBED_MODEL)

embedding_function = SentenceTransformerEmbeddings(
    model_name=EMBED_MODEL)

embedded_db = Chroma.from_documents(
    documents = all_chunks, 
    embedding = embedding_function, 
    persist_directory = LANGUAGE_CHROMO_PATH
)




current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
line_to_append = f"{current_time} - Embedding completed\n"
with open('embeddings_log.txt', 'a') as file:
    file.write(line_to_append)

print(f"Embedding to {LANGUAGE_CHROMO_PATH} complete.\n")




# texts = text_splitter.split_text(reports[0].page_content)
# text_docs_reports = text_splitter.split_documents(reports)
# text_docs2_chronicles = text_splitter.split_documents(control_chronicles)
# text_docs2_raynok = text_splitter.split_documents(raynok_report)
