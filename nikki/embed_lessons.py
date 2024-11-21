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

#system
import os, sys
import glob, argparse, yaml, json
import dotenv, re, datetime
dotenv.load_dotenv()


# Local Constants
from constants import EMBED_MODEL, LANGUAGE_LESSON_PATH, LANGUAGE_CHROMADB_PATH, LANGUAGE_LANCEDB_PATH
directory_path = LANGUAGE_LESSON_PATH


from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownTextSplitter

# Embeddings
from sentence_transformers import SentenceTransformer
from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_community.embeddings import SentenceTransformerEmbeddings

# Vector Store
from langchain_community.vectorstores import Chroma
# from langchain_community.vectorstores import LanceDB

# Documents
from itertools import islice
from langchain_community.document_loaders import TextLoader
from typing import List, Dict, Union
from collections import namedtuple


############################
# Helper Functions
############################

def get_file_paths(directory_path, patterns):
    files = []
    for pattern in patterns:
        files.extend(glob.glob(os.path.join(directory_path, pattern)))
    return files

def extract_metadata(content):
    # Extract top-level header
    header_match = re.search(r'^# (.+)$', content, re.MULTILINE)
    top_level_header = header_match.group(1) if header_match else "Unknown Lesson"
    
    # Extract lesson ID, language, and level
    lesson_id_match = re.search(r'\*\*Lesson ID:\*\* (\d+)', content)
    lesson_id = lesson_id_match.group(1) if lesson_id_match else "Unknown"
    
    return {
        "top_level_header": top_level_header,
        "lesson_id": lesson_id
    }

def read_markdown_files(directory):
    markdown_docs = []
    for filename in glob.glob(os.path.join(directory, 'ita*.md')):
        with open(filename, 'r', encoding='utf-8') as file:
            content = file.read()
            metadata = extract_metadata(content)
            markdown_docs.append((metadata, content))
    return markdown_docs


# Read the markdown files
markdown_docs = read_markdown_files(directory_path)
print(f"Processed {len(markdown_docs)} markdown documents.")



# Split into Chunks
markdown_splitter = MarkdownTextSplitter(
    chunk_size=2048, 
    chunk_overlap=64
    )

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1024, 
    chunk_overlap=128, 
    is_separator_regex=False
    )


# Process each markdown document
Document = namedtuple('Document', ['page_content', 'metadata'])
all_chunks = []
for metadata, markdown_content in markdown_docs:
    chunks = markdown_splitter.split_text(markdown_content)                                  
    doc_chunks = [
        Document(
            page_content=chunk,
            metadata=metadata
        ) for chunk in chunks
    ]
    all_chunks.extend(doc_chunks)

print("Chunk array size: ", len(all_chunks))


# Initialize the Sentence Transformer Model for Embeddings
# embeddings = SentenceTransformerEmbeddings(model_name=EMBED_MODEL)
embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL, model_kwargs={'device': 'cpu'})

query = "Show me the total number of embeddings"
print("Total embeddings for query: ", len(embeddings.embed_documents([query])[0]))



vectorstore = Chroma.from_documents(
    documents = all_chunks,
    embedding = embeddings,
    persist_directory = LANGUAGE_CHROMADB_PATH
)

#TODO: Upgrade to LanceDB
# vectorstore = LanceDB.from_documents(
#     documents = all_chunks,
#     embedding = embeddings,
#     persist_directory = LANGUAGE_LANCEDB_PATH
# )









# print("Verifying metadata for chunks:")
# for idx, doc in enumerate(all_chunks):
#     print(f"Chunk {idx}:")
#     print(f"Top-level Header: {doc.metadata['top_level_header']}")
#     print(f"Lesson ID: {doc.metadata['lesson_id']}")
#     print(f"Content preview: {doc.page_content[:50]}...")
#     print("---")

# Verify after persistence
# print("\nVerifying metadata in persisted vectorstore:")
# collection = vectorstore.get()
# for idx, doc in enumerate(collection['documents']):
#     print(f"Document {idx}:")
#     metadata = collection['metadatas'][idx]
    
#     top_level_header = metadata.get('top_level_header', 'N/A')
#     lesson_id = metadata.get('lesson_id', 'N/A')
#     language = metadata.get('language', 'N/A')
#     level = metadata.get('level', 'N/A')

#     print(f"Top-level Header: {top_level_header}")
#     print(f"lesson_id: {lesson_id}")
#     print(f"doc: {doc[:50]}...")
#     print("---")



current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
line_to_append = f"{current_time} - Successfully embedded vector db to {LANGUAGE_CHROMADB_PATH}\n"
with open('embeddings_log.txt', 'a') as file:
    file.write(line_to_append)

print(f"Embedding to {LANGUAGE_CHROMADB_PATH} complete.\n")




