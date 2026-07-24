
DB_DIR = "./chroma_db"
embedding_model_path="C:/offline_models/all-MiniLM-L6-v2"
llm_model="qwen3:1.7b"
OLLAMA_BASE_URL = "http://127.0.0.1:11434"

# List your available models here (synced with local Ollama instance)
MODEL_OPTIONS = [
    "qwen3:1.7b",
    "deepseek-r1:1.5b",
    "deepseek-r1:8b",
    "deepseek-r1:latest",
    "gemma3:1b",
    "gemma3:12b",
    "gemma3-16k:latest",
    "llama2:latest",
    "mistral:7b-instruct-v0.2-q4_K_M",
    "phi3:3.8b-mini-128k-instruct-q4_0",
    "tinyllama:latest",
    "gemma2:2b-text-q8_0",
    "gemma2:9b-instruct-q4_K_S",
    "qwen3-vl:8b",
    "qwen3-coder:30b",
    # Add more as needed
]