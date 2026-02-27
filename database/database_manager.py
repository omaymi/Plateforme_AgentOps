import sqlite3
import hashlib

class DatabaseManager:
    def __init__(self, db_name="agents_platform.db"):
        self.db_name = db_name
        self.init_db()

    def get_connection(self):
        conn = sqlite3.connect(self.db_name)
        # Active le support des clés étrangères (important pour SQLite)
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def init_db(self):
        """Initialise les tables si elles n'existent pas."""
        # On regroupe les créations de tables dans une seule fonction
        queries = [
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password TEXT NOT NULL
            );
            """,
            """
            CREATE TABLE IF NOT EXISTS agents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                name TEXT NOT NULL,
                model TEXT NOT NULL,
                vector_method TEXT NOT NULL,
                system_prompt TEXT,
                temperature REAL DEFAULT 0.7,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            );
            """
        ]
        with self.get_connection() as conn:
            for q in queries:
                conn.execute(q)
            conn.commit()
    def _migrate_db(self):
        """Ajoute la colonne user_id à la table agents si elle n'existe pas encore."""
        with self.get_connection() as conn:
            cursor = conn.execute("PRAGMA table_info(agents)")
            columns = [column[1] for column in cursor.fetchall()]
            
            if 'user_id' not in columns:
                print("Migration : Ajout de la colonne user_id à la table agents...")
                # On ajoute la colonne. On autorise NULL au début pour les anciens agents.
                conn.execute("ALTER TABLE agents ADD COLUMN user_id INTEGER REFERENCES users(id) ON DELETE CASCADE;")
                conn.commit()
    # --- GESTION DES UTILISATEURS ---

    def create_user(self, username, password):
        """Crée un utilisateur avec mot de passe haché."""
        hashed_pw = hashlib.sha256(password.encode()).hexdigest()
        query = "INSERT INTO users (username, password) VALUES (?, ?)"
        try:
            with self.get_connection() as conn:
                cursor = conn.execute(query, (username, hashed_pw))
                conn.commit()
                return cursor.lastrowid
        except sqlite3.IntegrityError:
            return None # Utilisateur déjà existant

    def verify_user(self, username, password):
        """Vérifie les identifiants et retourne l'utilisateur."""
        hashed_pw = hashlib.sha256(password.encode()).hexdigest()
        query = "SELECT id, username FROM users WHERE username = ? AND password = ?"
        with self.get_connection() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(query, (username, hashed_pw))
            row = cursor.fetchone()
            return dict(row) if row else None

    # --- GESTION DES AGENTS (ISOLÉE PAR USER) ---

    def create_agent(self, user_id, name, model, vector_method, system_prompt, temperature=0.7):
        """Crée un agent lié à un utilisateur spécifique."""
        query = """
        INSERT INTO agents (user_id, name, model, vector_method, system_prompt, temperature)
        VALUES (?, ?, ?, ?, ?, ?);
        """
        with self.get_connection() as conn:
            cursor = conn.execute(query, (user_id, name, model, vector_method, system_prompt, temperature))
            conn.commit()
            return cursor.lastrowid

    def get_user_agents(self, user_id):
        """Récupère uniquement les agents appartenant à l'utilisateur."""
        query = "SELECT * FROM agents WHERE user_id = ?;"
        with self.get_connection() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(query, (user_id,))
            return [dict(row) for row in cursor.fetchall()]

    def get_agent_by_id(self, agent_id, user_id):
        """Récupère un agent spécifique (vérifie qu'il appartient bien au user)."""
        query = "SELECT * FROM agents WHERE id = ? AND user_id = ?;"
        with self.get_connection() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(query, (agent_id, user_id))
            row = cursor.fetchone()
            return dict(row) if row else None

    def delete_agent(self, agent_id, user_id):
        """Supprime un agent si l'utilisateur en est le propriétaire."""
        query = "DELETE FROM agents WHERE id = ? AND user_id = ?;"
        with self.get_connection() as conn:
            cursor = conn.execute(query, (agent_id, user_id))
            conn.commit()
            return cursor.rowcount > 0 # Retourne True si supprimé

import sqlite3

def force_migrate():
    db_path = "agents_platform.db"
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # 1. Vérifier si la colonne existe déjà (sécurité)
    cursor.execute("PRAGMA table_info(agents)")
    columns = [column[1] for column in cursor.fetchall()]

    if 'user_id' not in columns:
        print("🔧 Ajout de la colonne 'user_id' en cours...")
        try:
            # 2. Ajouter la colonne
            cursor.execute("ALTER TABLE agents ADD COLUMN user_id INTEGER;")
            conn.commit()
            print("✅ Succès ! La colonne 'user_id' a été ajoutée.")
        except Exception as e:
            print(f"❌ Erreur lors de la migration : {e}")
    else:
        print("✅ La colonne 'user_id' existe déjà.")

    # 3. Vérification finale
    cursor.execute("PRAGMA table_info(agents)")
    print("\n--- Nouvelle Structure ---")
    for col in cursor.fetchall():
        print(f"Colonne: {col[1]} | Type: {col[2]}")
    
    conn.close()   
def check_structure():
    conn = sqlite3.connect("agents_platform.db")
    cursor = conn.execute("PRAGMA table_info(agents)")
    columns = cursor.fetchall()
    
    print("--- Structure de la table 'agents' ---")
    for col in columns:
        # col[1] est le nom de la colonne, col[2] est le type
        print(f"Colonne: {col[1]} | Type: {col[2]}")
    conn.close()



if __name__ == "__main__":
    force_migrate()
    check_structure()
# --- TEST RAPIDE ---
# if __name__ == "__main__":
#     db = DatabaseManager()
#     agent_id = db.create_agent(
#         name="Expert Sémantique",
#         model="mistral",
#         vector_method="sbert", # On utilise 'sbert' (ou tout nom différent de 'tfidf')
#         system_prompt="Tu es un assistant qui comprend les nuances du langage.",
#         temperature=0.4
#     )
#     print(f"Agent Sémantique créé avec l'ID : {agent_id}")       

    # import shutil

    # shutil.rmtree("db_vectors", ignore_errors=True)
    # shutil.rmtree("db_history", ignore_errors=True)     
                   

           

