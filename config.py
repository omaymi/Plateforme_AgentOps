import os

OLLAMA_HOST = "https://ollama-cipnmv-11434.svc-usw2.nicegpu.com/" 

# --- CONFIGURATION BASE DE DONNÉES ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_VECTORS_DIR = os.path.join(BASE_DIR, "db_vectors")
DB_HISTORY_DIR = os.path.join(BASE_DIR, "db_history")

# --- PARAMÈTRES IA PAR DÉFAUT ---
DEFAULT_MODEL = "llama3.1:8b" 
DEFAULT_TEMP = 0.7
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# --- LISTE DES MODÈLES DISPONIBLES (À METTRE À JOUR) ---
AVAILABLE_MODELS = [
    "llama3.1:8b",
    "mistral-nemo:12b",
    "gemma3:12b",
    "phi4:14b",
    "deepseek-r1"
]

# --- SÉCURITÉ (Optionnel) ---
# Vous pouvez ajouter ici des clés API si besoin plus tard