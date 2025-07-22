import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain_community.vectorstores import Pinecone as LangchainPinecone
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load environment variables
load_dotenv(".env.local")

# Environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_env = os.getenv("PINECONE_ENVIRONMENT")  # e.g., "us-east-1"
pinecone_index_name = os.getenv("PINECONE_INDEX")  # e.g., "atlasashok"

# Safety checks
assert openai_api_key, "❌ OPENAI_API_KEY not set"
assert pinecone_api_key, "❌ PINECONE_API_KEY not set"
assert pinecone_env, "❌ PINECONE_ENVIRONMENT not set"
assert pinecone_index_name, "❌ PINECONE_INDEX not set"

# Initialize Pinecone (new SDK)
pc = Pinecone(api_key=pinecone_api_key)
if pinecone_index_name not in [i.name for i in pc.list_indexes()]:
    pc.create_index(
        name=pinecone_index_name,
        dimension=1536,
        spec=ServerlessSpec(cloud="aws", region=pinecone_env)
    )
print(f"📦 Using Pinecone index: {pinecone_index_name}")

# Load DOCX files
doc_dir = "docs"
doc_paths = [os.path.join(doc_dir, f) for f in os.listdir(doc_dir) if f.endswith(".docx")]
print(f"📄 Found {len(doc_paths)} DOCX files")

all_docs = []
for path in doc_paths:
    print(f"📥 Loading: {path}")
    loader = UnstructuredFileLoader(path)
    all_docs.extend(loader.load())

# Chunk documents
text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
split_docs = text_splitter.split_documents(all_docs)
print(f"✂️ Split into {len(split_docs)} chunks")

# Embed and push to Pinecone
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
vectorstore = LangchainPinecone.from_documents(
    documents=split_docs,
    embedding=embeddings,
    index_name=pinecone_index_name,
    namespace="default"
)

print(f"✅ Uploaded {len(split_docs)} chunks to Pinecone index '{pinecone_index_name}'")
