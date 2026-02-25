import os
import streamlit as st
from groq import Groq
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.llms import MockLLM
import chromadb

load_dotenv()

Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5"
)
Settings.llm = None

chroma_client = chromadb.PersistentClient(path="./legal_db")
collection = chroma_client.get_or_create_collection("legal_knowledge")
vector_store = ChromaVectorStore(chroma_collection=collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_vector_store(vector_store)
query_engine = index.as_query_engine(similarity_top_k=3)

SYSTEM_PROMPT = """
You are a compassionate legal advisor for common people in India.
The person has NO legal background. They may be scared or confused.
Be their trusted guide like a knowledgeable friend who knows the law.

Always respond in this format:
- Empathy: acknowledge their situation warmly
- YOUR RIGHTS: what the law says in simple words
- WHAT TO DO NOW: numbered action steps
- DOCUMENTS TO COLLECT: specific list
- DO YOU NEED A LAWYER: honest yes or no with reason

Rules:
- Use simple everyday language, no legal jargon
- Never predict court outcomes
- Always end with this disclaimer:
  Disclaimer: This is general guidance only, not professional legal advice.
  Free legal aid available at your nearest DLSA or call 15100.
"""

st.set_page_config(
    page_title="Legal AI Advisor",
    page_icon="⚖️",
    layout="centered"
)

st.title("⚖️ Legal AI Advisor")
st.caption("Describe your situation and get free legal guidance in simple language.")

if "history" not in st.session_state:
    st.session_state.history = []

if "client" not in st.session_state:
    st.session_state.client = Groq(api_key=os.getenv("GROQ_API_KEY"))

for message in st.session_state.history:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if prompt := st.chat_input("Describe your legal situation here..."):
    st.session_state.history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing your situation..."):
            legal_context = str(query_engine.query(prompt))
            enhanced_prompt = SYSTEM_PROMPT + f"""

Relevant Law from Knowledge Base:
{legal_context}
"""
            response = st.session_state.client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": enhanced_prompt}
                ] + st.session_state.history,
                max_tokens=1500
            )
            reply = response.choices[0].message.content
            st.write(reply)

    st.session_state.history.append({"role": "assistant", "content": reply})