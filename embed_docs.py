import dotenv, os, file_helper
import nikki_templates

# Variable Loaders
from constants import MARKDOWN_REPORTS_PATH, ESTOP_LOG_PATH, REPORTS_CHROMA_PATH, EMBED_MODEL
dotenv.load_dotenv()

from langchain_community.vectorstores import Chroma
from sentence_transformers import SentenceTransformer
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

docs = file_helper.read_markdown_file(MARKDOWN_REPORTS_PATH)
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1024, 
    chunk_overlap=64, 
    is_separator_regex=False)

texts = text_splitter.split_text(docs[0].page_content)
text_docs = text_splitter.split_documents(docs)

# Initialize the Sentence Transformer Model for Embeddings
model = SentenceTransformer(EMBED_MODEL)
embedding_function = SentenceTransformerEmbeddings(model_name=EMBED_MODEL)
db2 = Chroma.from_documents(text_docs, embedding_function, persist_directory=REPORTS_CHROMA_PATH)

print("Embedding complete.\n")


