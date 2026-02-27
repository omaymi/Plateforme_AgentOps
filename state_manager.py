import os
import time
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from orchestrator import AgentOrchestrator
from ingestion import DocumentProcessor
from vector_engine import VectorEngine 
import config

class SessionState:
    def __init__(self, user_id=None):
        self.user_id = user_id
        self.current_agent_id = None
        self.orchestrator = None

        # Mémoire documents
        self.vector_db = None        # SBERT (persistant)
        self.memory_engine = None    # TF-IDF / CBOW (mémoire)

        # Mémoire conversationnelle
        self.history_db = None
        self.chat_history_buffer = []

        # Embedding (sera chargé seulement si nécessaire)
        self.embeddings = None

        # Dossiers persistants (uniquement pour Chroma)
        self.base_vector_dir = config.DB_VECTORS_DIR
        self.base_history_dir = config.DB_HISTORY_DIR

    def add_new_knowledge(self, chunks):
        """Ajoute dynamiquement des morceaux de texte au moteur de l'agent actif."""
        if not self.orchestrator:
            print("ERREUR - Aucun orchestrateur chargé pour l'ajout de connaissance.")
            return 

        method = self.orchestrator.agent_config.get('vector_method', 'tfidf')
        print(f"DEBUG - Indexation de {len(chunks)} fragments avec la méthode : {method}")

        if method == "sbert":
            if self.vector_db:
                self.vector_db.add_texts(chunks)
                # Chroma persiste automatiquement mais on peut forcer si besoin
            else:
                print("ERREUR - vector_db non initialisé pour SBERT.")
        else:
            if not self.memory_engine:
                self.memory_engine = VectorEngine([])
            
            self.memory_engine.documents.extend(chunks)
            if method == "tfidf":
                self.memory_engine.fit_tfidf()
            elif method == "cbow":
                self.memory_engine.fit_cbow()
            print(f"DEBUG - Index mis à jour. Total documents : {len(self.memory_engine.documents)}")

    def process_uploaded_file(self, uploaded_file):
        """Lit un fichier uploadé et l'ajoute directement à l'agent actif."""
        processor = DocumentProcessor()
        file_bytes = uploaded_file.getvalue()

        if uploaded_file.type == "application/pdf" or uploaded_file.name.endswith(".pdf"):
            text = processor.process_uploaded_file(file_bytes)
        else:
            text = file_bytes.decode("utf-8")

        chunks = processor.chunk_text(text)
        self.add_new_knowledge(chunks) # Utilise la méthode centralisée 


    # --------------------------------------------------
    # CHARGEMENT AGENT
    # --------------------------------------------------

    def load_agent(self, agent_id):
        if self.current_agent_id != agent_id:
            self.unload_current_agent()
            self.current_agent_id = agent_id
            self.orchestrator = AgentOrchestrator(agent_id, self.user_id)

            method = self.orchestrator.agent_config['vector_method']
            self.agent_vector_dir = os.path.join(self.base_vector_dir, f"user_{self.user_id}", f"agent_{agent_id}")
            self.agent_history_dir = os.path.join(self.base_history_dir, f"user_{self.user_id}", f"agent_{agent_id}")

            if method == "sbert":
                self.embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
                self.vector_db = Chroma(embedding_function=self.embeddings, persist_directory=self.agent_vector_dir)
            else:
                self.memory_engine = VectorEngine([])

            self._init_history_db()

    def unload_current_agent(self):
        self.current_agent_id = None
        self.orchestrator = None
        self.vector_db = None
        self.memory_engine = None
        self.history_db = None
        self.embeddings = None
        self.chat_history_buffer = []
        import gc
        gc.collect()

    def _init_history_db(self):
        if not self.embeddings:
            self.embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        self.history_db = Chroma(embedding_function=self.embeddings, persist_directory=self.agent_history_dir)

    # --------------------------------------------------
    # QUESTION
    # --------------------------------------------------

    def ask_question(self, question):

        if not self.orchestrator:
            return "Chargez un agent d'abord.", None

        method = self.orchestrator.agent_config['vector_method']

        history_context = "\n".join(self.chat_history_buffer[-5:])

        doc_context = ""

        # --- Recherche documents ---
        if method == "sbert" and self.vector_db:
            docs = self.vector_db.similarity_search(question, k=3)
            print("DEBUG - Documents récupérés depuis Chroma:")
            doc_context = "\n\n".join([d.page_content for d in docs])

        elif method in ["tfidf", "cbow"] and self.memory_engine:
            results = self.memory_engine.search(
                question,
                method=method,
                top_k=3
            )
            doc_context = "\n\n".join(results)

        else:
            doc_context = "Pas de document."

        # --- Recherche mémoire long terme ---
        long_term_mem = ""
        if self.history_db:
            mem_docs = self.history_db.similarity_search(question, k=2)
            if mem_docs:
                long_term_mem = "\n[Faits anciens] : " + " | ".join(
                    [d.page_content for d in mem_docs]
                )

        full_context = f"{doc_context}\n{long_term_mem}"

        response = self.orchestrator.generate_response(
            question,
            full_context,
            history_context
        )

        self.chat_history_buffer.append(f"Utilisateur: {question}")
        self.chat_history_buffer.append(f"Assistant: {response}")

        self._save_to_history(question, response)

        return response, full_context


    # --------------------------------------------------
    # SAUVEGARDE HISTORIQUE
    # --------------------------------------------------

    def _save_to_history(self, question, response):

        if self.history_db:
            interaction = f"Q: {question} | R: {response}"

            self.history_db.add_texts(
                texts=[interaction],
                metadatas=[{"timestamp": time.time()}]
            )