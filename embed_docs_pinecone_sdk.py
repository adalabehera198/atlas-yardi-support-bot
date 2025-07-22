import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load .env.local
load_dotenv(".env.local")

# Environment vars
openai_api_key = os.getenv("OPENAI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_env = os.getenv("PINECONE_ENVIRONMENT")
pinecone_index_name = os.getenv("PINECONE_INDEX")

assert openai_api_key, "❌ OPENAI_API_KEY missing"
assert pinecone_api_key, "❌ PINECONE_API_KEY missing"
assert pinecone_env, "❌ PINECONE_ENVIRONMENT missing"
assert pinecone_index_name, "❌ PINECONE_INDEX missing"

# Init Pinecone v3
pc = Pinecone(api_key=pinecone_api_key)
if pinecone_index_name not in [i.name for i in pc.list_indexes()]:
    pc.create_index(
        name=pinecone_index_name,
        dimension=1536,
        spec=ServerlessSpec(cloud="aws", region=pinecone_env)
    )
index = pc.Index(pinecone_index_name)

# Load DOCX files
doc_dir = "docs"
doc_paths = [os.path.join(doc_dir, f) for f in os.listdir(doc_dir) if f.endswith(".docx")]
print(f"📄 Found {len(doc_paths)} DOCX files")

all_docs = []
for path in doc_paths:
    print(f"📥 Loading: {path}")
    loader = UnstructuredFileLoader(path)
    all_docs.extend(loader.load())

# Chunking
text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
split_docs = text_splitter.split_documents(all_docs)
print(f"✂️ Split into {len(split_docs)} chunks")

# Embeddings
embedder = OpenAIEmbeddings(openai_api_key=openai_api_key)
vectors = []
for i, doc in enumerate(split_docs):
    embedding = embedder.embed_query(doc.page_content)
    vectors.append({
        "id": f"chunk-{i}",
        "values": embedding,
        "metadata": {"text": doc.page_content[:100]}
    })

# Upsert
index.upsert(vectors=vectors, namespace="default")
print(f"✅ Uploaded {len(vectors)} chunks to Pinecone '{pinecone_index_name}' (v3 SDK)")
