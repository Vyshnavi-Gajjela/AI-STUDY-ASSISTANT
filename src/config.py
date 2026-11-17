from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOCUMENTS_DIR = PROJECT_ROOT / "data" / "documents"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

# Maximum number of passages sent to the LLM.
TOP_K = 10

# Number of passages considered by semantic retrieval before fusion.
RETRIEVAL_CANDIDATES = 30

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
LLM_MODEL_NAME = "llama3.2:3b"
OLLAMA_URL = "http://localhost:11434/api/generate"