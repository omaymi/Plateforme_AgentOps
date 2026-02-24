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
        
        # Dossiers de base
        self.base_vector_dir = "db_vectors"
        self.base_history_dir = "db_history"
        self.base_knowledge_dir = "knowledge"
        
        self.chat_history_buffer = [] 

    def load_agent(self, agent_id):
        if self.current_agent_id != agent_id:
            # On décharge l'ancien avant de charger le nouveau
            self.unload_current_agent()
            
            self.current_agent_id = agent_id
            self.orchestrator = AgentOrchestrator(agent_id)
            
            # 1. Isolation des chemins selon l'agent
            self.agent_vector_dir = os.path.join(self.base_vector_dir, f"agent_{agent_id}")
            self.agent_history_dir = os.path.join(self.base_history_dir, f"agent_{agent_id}")
            self.agent_knowledge_dir = os.path.join(self.base_knowledge_dir, f"agent_{agent_id}")
            
            # Création des dossiers si nécessaire
            os.makedirs(self.agent_knowledge_dir, exist_ok=True)
            
            # 2. Initialisation de l'historique isolé
            self._init_history_db()
            
            # 3. Chargement automatique de la connaissance permanente
            self._load_permanent_knowledge()

    def unload_current_agent(self):
        """Libère explicitement les ressources pour éviter les verrous de fichiers sur Windows."""
        self.current_agent_id = None
        self.orchestrator = None
        self.vector_db = None       
        self.history_db = None      
        self.memory_engine = None   
        self.documents = []
        self.chat_history_buffer = []
        # Petit délai pour laisser le temps au garbage collector si nécessaire
        import gc
        gc.collect()

    def _init_history_db(self):
        self.history_db = Chroma(
            embedding_function=self.embeddings,
            persist_directory=self.agent_history_dir
        )

    def _load_permanent_knowledge(self):
        """Charge tous les fichiers présents dans le dossier de connaissance de l'agent."""
        files = [os.path.join(self.agent_knowledge_dir, f) for f in os.listdir(self.agent_knowledge_dir) 
                 if f.endswith(('.pdf', '.txt'))]
        
        if files:
            # On réinitialise les documents et on les traite un par un
            self.documents = []
            processor = DocumentProcessor()
            for file_path in files:
                self.documents.extend(processor.process_file(file_path))
            
            # On (re)construit l'index vectoriel
            self._build_vector_index()

    def _build_vector_index(self):
        """Construit l'index (TF-IDF ou Chroma) basé sur self.documents."""
        if not self.documents:
            return

        method = self.orchestrator.agent_config['vector_method']
        if method == "tfidf":
            self.memory_engine = VectorEngine(self.documents)
            self.memory_engine.fit_tfidf()
        else:
            self.vector_db = Chroma.from_texts(
                texts=self.documents,
                embedding=self.embeddings,
                persist_directory=self.agent_vector_dir
            )

    def process_document(self, file_path):
        """Gère l'upload à la volée pendant la discussion."""
        # On déplace le fichier dans le dossier permanent de l'agent pour qu'il soit retenu
        dest_path = os.path.join(self.agent_knowledge_dir, os.path.basename(file_path))
        if file_path != dest_path:
            import shutil
            shutil.move(file_path, dest_path)
        
        # On recharge toute la connaissance pour inclure le nouveau fichier
        self._load_permanent_knowledge()

    def ask_question(self, question):
        if not self.orchestrator:
            return "Erreur : Chargez un agent d'abord.", None

        method = self.orchestrator.agent_config['vector_method']

        # --- 1. PRÉPARATION DE LA MÉMOIRE COURTE (Buffer) ---
        history_context = "\n".join(self.chat_history_buffer[-5:])

        # --- 2. RECHERCHE DE CONTEXTE DOCS ---
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

        # --- 3. RECHERCHE MÉMOIRE LONG TERME ---
        long_term_mem = ""
        if self.history_db:
            try:
                mem_docs = self.history_db.similarity_search(question, k=2)
                if mem_docs:
                    long_term_mem = "\n[Faits anciens] : " + " | ".join([d.page_content for d in mem_docs])
            except Exception:
                # Gérer le cas où la DB est vide au début
                pass

        full_context = f"{doc_context}\n{long_term_mem}"

        # --- 4. GÉNÉRATION ---
        response = self.orchestrator.generate_response(question, full_context, history_context)

        # --- 5. SAUVEGARDE DOUBLE ---
        self.chat_history_buffer.append(f"Utilisateur: {question}")
        self.chat_history_buffer.append(f"Assistant: {response}")
        self._save_to_history(question, response)
        
        return response, full_context

    def _save_to_history(self, question, response):
        if self.history_db:
            interaction = f"Q: {question} | R: {response}"
            self.history_db.add_texts(
                texts=[interaction],
                metadatas=[{"timestamp": time.time()}]
            )
