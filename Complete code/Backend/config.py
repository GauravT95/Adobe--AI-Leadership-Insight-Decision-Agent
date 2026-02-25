"""
Configuration for Azure OpenAI and application settings.
Update these values to match your Azure deployment.
"""
import os

# ─── Azure OpenAI ───

## PLEASE PASS YOUR KEY HERE
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "https://ue2daoipocaoa0l.openai.azure.com/")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
AZURE_CHAT_MODEL = os.getenv("AZURE_CHAT_MODEL", "gpt-5")
AZURE_EMBEDDING_MODEL = os.getenv("AZURE_EMBEDDING_MODEL", "text-embedding")

# ─── RAG Settings ───
CHUNK_SIZE = 350
CHUNK_OVERLAP = 100
TOP_K = 5
CACHE_SIMILARITY_THRESHOLD = 0.92
CACHE_MAX_SIZE = 50

# ─── Document Directory ───
DOC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample_company_docs")
