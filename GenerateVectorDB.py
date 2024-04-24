from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings

load_dotenv()

# embeddings = OllamaEmbeddings()
embeddings = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2")

textSplitter = CharacterTextSplitter(
    separator="--n",
    chunk_size = 170,
    chunk_overlap = 0
)

loader = TextLoader('PersonDescription.txt')
docs = loader.load_and_split(
    text_splitter=textSplitter
)

db = Chroma.from_documents(
    docs,
    embedding=embeddings,
    persist_directory="emb"
)
# db = Chroma(
#     embedding_function=embeddings,
#     persist_directory="emb"
# )

# results = db.similarity_search_with_score("What are some of his projects", k=2)

# for result in results:
#     print("\n")
#     print(result[0].page_content)