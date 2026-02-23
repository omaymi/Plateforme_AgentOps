import os
import time
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from orchestrator import AgentOrchestrator
from ingestion import DocumentProcessor
from vector_engine import VectorEngine 

class SessionState:
    def __init__(self):
        self.current_agent_id = None
        self.orchestrator = None
        self.vector_db = None       
        self.history_db = None      
        self.memory_engine = None   
        self.documents = []         
        self.embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        self.persist_directory = "db_vectors"
        self.history_directory = "db_history"
        # NOUVEAU : Buffer de mémoire immédiate (court terme)
        self.chat_history_buffer = [] 

    def load_agent(self, agent_id):
        if self.current_agent_id != agent_id:
            self.current_agent_id = agent_id
            self.orchestrator = AgentOrchestrator(agent_id)
            self._init_history_db(agent_id)
            # On vide le buffer court terme si on change d'agent
            self.chat_history_buffer = []

    def _init_history_db(self, agent_id):
        path = f"{self.history_directory}_agent_{agent_id}"
        self.history_db = Chroma(
            embedding_function=self.embeddings,
            persist_directory=path
        )

    def process_document(self, file_path):
        processor = DocumentProcessor()
        self.documents = processor.process_file(file_path)
        method = self.orchestrator.agent_config['vector_method']
        if method == "tfidf":
            self.memory_engine = VectorEngine(self.documents)
            self.memory_engine.fit_tfidf()
        else:
            self.vector_db = Chroma.from_texts(
                texts=self.documents,
                embedding=self.embeddings,
                persist_directory=self.persist_directory
            )

    def ask_question(self, question):
        if not self.orchestrator:
            return "Erreur : Chargez un agent d'abord.", None

        method = self.orchestrator.agent_config['vector_method']

        # --- 1. PRÉPARATION DE LA MÉMOIRE COURTE (Buffer) ---
        # On prend les 5 derniers messages pour le contexte immédiat
        history_context = "\n".join(self.chat_history_buffer[-5:])

        # --- 2. RECHERCHE DE CONTEXTE DOCS (PDF) ---
        doc_context = ""
        has_docs = (method == "tfidf" and self.memory_engine) or (self.vector_db is not None)
        if has_docs:
            if method == "tfidf":
                doc_context = self.memory_engine.search(question, method="tfidf")
            else:
                docs = self.vector_db.similarity_search(question, k=3)
                doc_context = "\n\n".join([d.page_content for d in docs])
        else:
            doc_context = "Pas de document. Réponds avec tes connaissances."

        # --- 3. RECHERCHE MÉMOIRE LONG TERME (ChromaDB history) ---
        long_term_mem = ""
        if self.history_db:
            mem_docs = self.history_db.similarity_search(question, k=2)
            if mem_docs:
                long_term_mem = "\n[Faits anciens] : " + " | ".join([d.page_content for d in mem_docs])

        # Fusion des contextes pour l'orchestrateur
        full_context = f"{doc_context}\n{long_term_mem}"

        # --- 4. GÉNÉRATION (On passe l'historique court + le contexte docs) ---
        response = self.orchestrator.generate_response(question, full_context, history_context)

        # --- 5. SAUVEGARDE DOUBLE (Court terme + Long terme) ---
        # Court terme (Buffer RAM)
        self.chat_history_buffer.append(f"Utilisateur: {question}")
        self.chat_history_buffer.append(f"Assistant: {response}")
        
        # Long terme (Vector DB Disque)
        self._save_to_history(question, response)
        
        return response, full_context

    def _save_to_history(self, question, response):
        if self.history_db:
            interaction = f"Q: {question} | R: {response}"
            self.history_db.add_texts(
                texts=[interaction],
                metadatas=[{"timestamp": time.time()}]
            )