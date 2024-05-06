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
from constants import MARKDOWN_REPORTS_PATH, LANGUAGE_CHROMO_PATH, EMBED_MODEL, LANGUAGE_LESSON_01
import dotenv, os, file_helper
import glob, json
dotenv.load_dotenv()

# Vector Store
from langchain_community.vectorstores import Chroma
from sentence_transformers import SentenceTransformer
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Local Imports
import template_nikki as template_nikki

# Load Language Lesson Markdown Files
directory_path = LANGUAGE_LESSON_01
pattern = os.path.join(directory_path, "ita_101_01*.json")
file_list = glob.glob(pattern)
language_docs = []
for file_path in file_list:
    with open(file_path, 'r', encoding='utf-8') as file:
        # Load the file as a JSON object
        data = json.load(file)
        
        # Assuming the text is stored under a key named 'text', modify as needed
        text_content = data['text']
        
        # Append the extracted text content to the language_docs list
        language_docs.append(text_content)

# Read files
# reports = file_helper.read_markdown_file(MARKDOWN_REPORTS_PATH)

# Split into Chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1024, 
    chunk_overlap=64, 
    is_separator_regex=False
)
language_text_split = text_splitter.split_documents(language_docs)

docs_to_embed = language_text_split

# Initialize the Sentence Transformer Model for Embeddings
model = SentenceTransformer(EMBED_MODEL)
embedding_function = SentenceTransformerEmbeddings(model_name=EMBED_MODEL)
db2 = Chroma.from_documents(docs_to_embed, embedding_function, persist_directory=LANGUAGE_CHROMO_PATH)

print("Embedding complete.\n")





# texts = text_splitter.split_text(reports[0].page_content)
# text_docs_reports = text_splitter.split_documents(reports)
# text_docs2_chronicles = text_splitter.split_documents(control_chronicles)
# text_docs2_raynok = text_splitter.split_documents(raynok_report)
