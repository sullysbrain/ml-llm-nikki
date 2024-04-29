import dotenv
import nikki_templates
import file_helper

# Variable Loaders
from constants import MARKDOWN_REPORTS_PATH, ESTOP_LOG_PATH, REPORTS_CHROMA_PATH
from _private.api import API_KEY_OPENAI
dotenv.load_dotenv()

from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings




docs = file_helper.read_markdown_file(MARKDOWN_REPORTS_PATH)
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1024, 
    chunk_overlap=64, 
    is_separator_regex=False)

texts = text_splitter.split_text(docs[0].page_content)
report_docs = ' '.join(texts)
 
reports_vector_db = Chroma.from_texts(
    texts, OpenAIEmbeddings(api_key=API_KEY_OPENAI), persist_directory=REPORTS_CHROMA_PATH
)
