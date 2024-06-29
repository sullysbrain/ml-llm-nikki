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
import glob, json
dotenv.load_dotenv()

# Vector Store
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from sentence_transformers import SentenceTransformer

from chromadb.utils import embedding_functions
from collections import namedtuple

# Load Docments to Embed
directory_path = "./_private/reports/"
pattern = os.path.join(directory_path, "report_*.md")
file_list = glob.glob(pattern)
# file_to_read = "./_private/reports/reports_all.md"

text_data_list = []

# Function to read Markdown files and extract text content
def read_markdown_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()


Document = namedtuple('Document', ['page_content', 'metadata'])

for file_path in file_list:
    reports = read_markdown_file(file_path)

    text_data_list.append(Document(page_content=reports, metadata={}))

    # text_content = read_markdown_file(file_path)
    # combined_text += text_content + "\n"  # Adding newline for separation
    # formatted_text= text_content[Document(page_content=text_content, metadata={})]
    # combined_text += text_content + "\n"  # Adding newline for separation
    # with open(file_path, 'r', encoding='utf-8') as file:
        # text_content = file.read()
    # text_data_list.append(text_content)




# text_content = ""
# with open(file_to_read, 'r', encoding='utf-8') as file:
#     text_content = file.read()


print(text_data_list)
print("\n\n")


# Split into Chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1024, 
    chunk_overlap=64, 
    is_separator_regex=False
)

# report_text_split = text_splitter.split_documents(text_data_list)
report_text_split = text_splitter.split_documents(text_data_list)

docs_to_embed = report_text_split

# Initialize the Sentence Transformer Model for Embeddings
model = SentenceTransformer(EMBED_MODEL)
embedding_function = SentenceTransformerEmbeddings(model_name=EMBED_MODEL)

db2 = Chroma.from_documents(docs_to_embed, embedding_function, persist_directory=REPORTS_CHROMA_PATH)

# db2 = Chroma.from_texts(docs_to_embed, embedding_function, persist_directory=REPORTS_CHROMA_PATH)

print("Embedding complete.\n")





# texts = text_splitter.split_text(reports[0].page_content)
# text_docs_reports = text_splitter.split_documents(reports)
# text_docs2_chronicles = text_splitter.split_documents(control_chronicles)
# text_docs2_raynok = text_splitter.split_documents(raynok_report)
