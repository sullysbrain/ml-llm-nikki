import dotenv, os, file_helper
import nikki_templates

# Variable Loaders
from constants import MARKDOWN_REPORTS_PATH, ESTOP_LOG_PATH, REPORTS_CHROMA_PATH, MARKDOWN_CONTROL_PATH, EMBED_MODEL
dotenv.load_dotenv()

from langchain_community.vectorstores import Chroma
from sentence_transformers import SentenceTransformer
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

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


