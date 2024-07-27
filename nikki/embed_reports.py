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
from constants import LANGUAGE_LESSON_PATH, LANGUAGE_CHROMO_PATH, EMBED_MODEL, REPORTS_CHROMA_PATH, REPORTS_PATH
import dotenv, re, datetime
dotenv.load_dotenv()

import os, glob, argparse, yaml, json

# Vector Store
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.document_loaders import TextLoader
from typing import List, Dict, Union

from collections import namedtuple


# LOAD ITALIAN CLASSES OR REPORTS
load_options_list = [
    [LANGUAGE_CHROMO_PATH, LANGUAGE_LESSON_PATH,['ita_*.yaml']],
    [REPORTS_CHROMA_PATH, REPORTS_PATH,['report_*.md']]
]
LOAD_ITALIAN = 0
LOAD_REPORTS = 1


# CLI Parser for Embed_docs
# usage: to load reports, type in CLI:  python embed_docs.py reports
# usage: to load tutor, type in CLI:  python embed_docs.py tutor
parser = argparse.ArgumentParser()
parser.add_argument("docs")
args = parser.parse_args()

target_docs = args.docs

if target_docs == "":
    print("The target docs don't exist.")
    raise SystemExit(1)
elif target_docs == "italian":
    report_to_load = LOAD_ITALIAN
elif target_docs == "reports":
    report_to_load = LOAD_REPORTS
else:
    report_to_load = LOAD_REPORTS


# Load documents based on CLI input
CHROMA_PATH = load_options_list[report_to_load][0]
DATA_PATH = load_options_list[report_to_load][1]
data_patterns = load_options_list[report_to_load][2]


## Load Documents For Embedding

if report_to_load == LOAD_ITALIAN:
    loaded_docs = "Italian Lessons"
else:
    loaded_docs = "Reports"



# Reads YAML file, returns data as json
# def read_yaml_file_as_json(file_path):
#     with open(file_path, 'r') as yaml_in:
#         data = yaml.safe_load(yaml_in)
#         json_data = json.dumps(data, indent=2)
#         return json_data

def get_file_paths(directory_path, patterns):
    files = []
    for pattern in patterns:
        files.extend(glob.glob(os.path.join(directory_path, pattern)))
    return files

def extract_date(filename):
    # Regular expression pattern to match the date at the end of the filename
    pattern = r'_(\d{8})\.md$'
    match = re.search(pattern, filename)
    
    # If a match is found, return the date string
    if match:
        return match.group(1)
    else:
        return None

Document = namedtuple('Document', ['page_content', 'metadata'])

def process_report(file, data) -> List[Dict]:
    documents = []
    documents.append({
        "content": text,
        "metadata": {
            "path": file,
            "date": extract_date(file)
        }
    })
    return documents

def process_lesson(file, data: Dict) -> List[Dict]:
    documents = []
    lesson_number = str(data.get("LessonNumber", "Unknown"))
    
    def recurse(current_data, current_prefix):
        if isinstance(current_data, dict):
            for key, value in current_data.items():
                new_prefix = f"{current_prefix} {key}" if current_prefix else key
                recurse(value, new_prefix)
        elif isinstance(current_data, list):
            for i, item in enumerate(current_data):
                new_prefix = f"{current_prefix} {i}" if current_prefix else str(i)
                recurse(item, new_prefix)
        else:
            documents.append({
                "content": str(current_data),
                "metadata": {
                    "path": file,
                    "lesson": lesson_number
                }
            })

    recurse(data, "")
    return documents



# Load Docments to Embed
files_to_read = get_file_paths(DATA_PATH, data_patterns)

isFirstReport = True

all_docs = []
for file in files_to_read:
    if report_to_load == LOAD_ITALIAN:
        with open(file, 'r') as f:
            data = yaml.safe_load(f)
        all_docs.extend(process_lesson(file, data))
    else: ## Load Reports
        with open(file, 'r') as f:
            text = f.read()
        all_docs.extend(process_report(file, text))
        # all_docs.append(Document(
        #     content=text,
        #     metadata={
        #         "path": file,
        #         "date": report_date
        #     }
        # ))

## Sort all_docs by filename
all_docs = sorted(all_docs, key=lambda x: x["metadata"]["path"])

## print the date of each doc in all_docs
for doc in all_docs:
    # perform switch case on report_to_load
    print(f"Lodaed Doc: {doc['metadata']['path']}")




# Split into Chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1024, 
    chunk_overlap=64, 
    is_separator_regex=False
)
split_docs = []
for doc in all_docs:
    splits = text_splitter.split_text(doc["content"])
    for split in splits:
        split_docs.append(Document(
            page_content=split,
            metadata=doc["metadata"]  # Maintain the original metadata for each split
        ))
        # print(f"Split: {split}\tMetadata: {doc['metadata']}\n")



# Initialize the Sentence Transformer Model for Embeddings
model = SentenceTransformer(EMBED_MODEL)

embedding_function = SentenceTransformerEmbeddings(
    model_name=EMBED_MODEL)

embedded_db = Chroma.from_documents(
    split_docs, 
    embedding_function, 
    persist_directory = CHROMA_PATH
)




current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
line_to_append = f"{current_time} - Embedding {loaded_docs} completed\n"
with open('embeddings_log.txt', 'a') as file:
    file.write(line_to_append)

print(f"Embedding {loaded_docs} to {CHROMA_PATH} complete.\n")




# texts = text_splitter.split_text(reports[0].page_content)
# text_docs_reports = text_splitter.split_documents(reports)
# text_docs2_chronicles = text_splitter.split_documents(control_chronicles)
# text_docs2_raynok = text_splitter.split_documents(raynok_report)
