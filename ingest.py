import os
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import chromadb

# Use free local embedding model — no API key needed!
Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5"
)

print("Loading legal documents...")
documents = SimpleDirectoryReader("./legal_docs").load_data()
print(f"Loaded {len(documents)} document chunks!")

print("Setting up database...")
chroma_client = chromadb.PersistentClient(path="./legal_db")
collection = chroma_client.get_or_create_collection("legal_knowledge")
vector_store = ChromaVectorStore(chroma_collection=collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

print("Storing documents in database...")
index = VectorStoreIndex.from_documents(
    documents,
    storage_context=storage_context
)

print("Done! Legal knowledge base is ready!")