from langchain_community.document_loaders.markdown import UnstructuredMarkdownLoader
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

import csv
#import os



# Read CSV file content
def read_csv_file(file_path):
    csv_texts = []
    with open(file_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            # Assuming each row is a list of columns and you concatenate them into one string per row
            csv_texts.append(' '.join(row))
    return csv_texts


def read_markdown_file(file):
    loader = UnstructuredMarkdownLoader(file)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=64, is_separator_regex=False)
    texts = text_splitter.split_text(docs[0].page_content)
    # Convert texts list into a single string of all items
    report_docs = ' '.join(texts)
    return report_docs



