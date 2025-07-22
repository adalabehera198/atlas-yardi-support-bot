import os
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone as LangchainPinecone
from langchain.document_loaders import UnstructuredFileLoader

# Load environment variables
load_dotenv(".env.local")

# Get keys
openai_api_key = os.getenv("OPENAI_API_KEY")
pinecone_index_name = os.getenv("PINECONE_INDEX")

assert openai_api_key, "❌ OPENAI_API_KEY not set"
assert pinecone_index_name, "❌ PINECONE_INDEX not set"

# Load .docx documents
doc_dir = "docs"
doc_paths = [os.path.join(doc_dir, f) for f in os.listdir(doc_dir) if f.endswith(".docx")]

print(f"📄 Found {len(doc_paths)} DOCX files")

all_docs = []
for path in doc_paths:
    print(f"📥 Loading: {path}")
    loader = UnstructuredFileLoader(path)
    all_docs.extend(loader.load())

# Split into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
split_docs = text_splitter.split_documents(all_docs)
print(f"✂️ Split into {len(split_docs)} chunks")

# Embed and push
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
vectorstore = LangchainPinecone.from_documents(
    documents=split_docs,
    embedding=embeddings,
    index_name=pinecone_index_name,
    namespace="default"
)

print(f"✅ Uploaded {len(split_docs)} chunks to Pinecone index '{pinecone_index_name}'")
