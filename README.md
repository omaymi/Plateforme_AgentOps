# app.py                 # Point d'entrée Streamlit (UI)
# orchestrator.py        # Logique de génération LLM (LangChain)
# state_manager.py       # Gestion de l'état et de la session
# ingestion.py           # Chargement et découpage (PDF/TXT)
# vector_engine.py       # le moteur manuel (TF-IDF/CBOW)
# database/database_manager.py # Gestion de la base de données SQLite
# test_connection.py     # Script de test de connexion à Ollama
# agents_platform.db  → configuration agents sqlite
# db_vectors/         → embeddings documents
# db_history/         → mémoire conversationnelle
