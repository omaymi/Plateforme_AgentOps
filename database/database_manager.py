import sqlite3

class DatabaseManager:
    def __init__(self, db_name="agents_platform.db"):
        self.db_name=db_name
        self.init_db()

    def get_connection(self):
        return sqlite3.connect(self.db_name)  
    def init_db(self):
        query = """
        CREATE TABLE IF NOT EXISTS agents (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        model TEXT NOT NULL,
        vector_method TEXT NOT NULL,
        system_prompt TEXT,
        temperature REAL DEFAULT 0.7
        );
        """
        with self.get_connection() as conn:
            conn.execute(query)
            conn.commit()
    def create_agent(self, name, model, vector_method, system_prompt, temperature=0.7):
        query= """
        INSERT INTO agents (name, model ,vector_method, system_prompt, temperature)
        VALUES (?,?,?,?,?);
        """
        with self.get_connection() as conn:
            cursor = conn.execute(query, (name, model, vector_method, system_prompt, temperature))
            conn.commit()
            return cursor.lastrowid

    def get_all_agents(self):
        """Récupère la liste de tous les agents."""
        query = "SELECT * FROM agents;"
        with self.get_connection() as conn:
            # Pour avoir les résultats sous forme de dictionnaire
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(query)
            return [dict(row) for row in cursor.fetchall()] 
    def get_agent_by_id(self, agent_id):
        """Récupère la configuration d'un agent spécifique."""
        query = "SELECT * FROM agents WHERE id = ?;"
        with self.get_connection() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(query, (agent_id,))
            row = cursor.fetchone()
            return dict(row) if row else None       

# --- TEST RAPIDE ---
if __name__ == "__main__":
    db = DatabaseManager()
    agent_id = db.create_agent(
        name="Expert Sémantique",
        model="mistral",
        vector_method="sbert", # On utilise 'sbert' (ou tout nom différent de 'tfidf')
        system_prompt="Tu es un assistant qui comprend les nuances du langage.",
        temperature=0.4
    )
    print(f"Agent Sémantique créé avec l'ID : {agent_id}")       
                   

           

