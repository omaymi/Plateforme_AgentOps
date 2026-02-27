import streamlit as st
import os
import re
import shutil
import time
from state_manager import SessionState
from database.database_manager import DatabaseManager
from ingestion import DocumentProcessor

# --- 1. CONFIGURATION & STYLE ---
st.set_page_config(page_title="AgentOps Factory", page_icon="⚡", layout="wide")

def apply_custom_style():
    st.markdown("""
        <style>
        /* Design des onglets */
        button[data-baseweb="tab"] { flex-grow: 1; justify-content: center; }
        div[data-baseweb="tab-list"] { gap: 20px; width: 100%; display: flex; justify-content: center; }
        button[data-baseweb="tab"] div p { font-size: 22px !important; font-weight: 600 !important; padding: 10px 20px; }
        div[data-baseweb="tab-highlight"] { background-color: #238636 !important; height: 4px !important; }

        /* Dark Mode Modern Look */
        .stApp { background: linear-gradient(160deg, #0e1117 0%, #161b22 100%); }
        div[data-baseweb="tab-panel"] {
            background-color: #1c2128; padding: 2rem; border-radius: 15px;
            border: 1px solid #30363d; box-shadow: 0 4px 12px rgba(0,0,0,0.3);
        }
        section[data-testid="stSidebar"] { background-color: #0d1117; border-right: 1px solid #30363d; }
        
        /* Boutons stylisés */
        .stButton>button { border-radius: 8px; transition: all 0.3s ease; background-color: #238636; color: white; border: none; }
        .stButton>button:hover { transform: translateY(-2px); box-shadow: 0 5px 15px rgba(35, 134, 54, 0.4); }
        
        /* Style spécifique pour le container d'authentification */
        .auth-box {
            background-color: #1c2128;
            padding: 30px;
            border-radius: 15px;
            border: 1px solid #30363d;
            text-align: center;
        }
        </style>
        """, unsafe_allow_html=True)

apply_custom_style()

# --- 2. INITIALISATION ---
db = DatabaseManager()

if "user" not in st.session_state:
    st.session_state.user = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "app_state" not in st.session_state or not hasattr(st.session_state.app_state, 'add_new_knowledge'):
    st.session_state.app_state = SessionState(user_id=None)

# --- 3. LOGIQUE D'AUTHENTIFICATION ---
def show_auth_page():
    st.markdown("<br><br>", unsafe_allow_html=True)
    _, col, _ = st.columns([1, 1.5, 1])
    
    with col:
        st.image("https://cdn-icons-png.flaticon.com/512/2103/2103633.png", width=80)
        st.title("AgentOps **Factory**")
        
        mode = st.tabs(["Connexion", "Créer un compte"])
        
        # Onglet Connexion
        with mode[0]:
            with st.form("login_form"):
                u_log = st.text_input("Nom d'utilisateur")
                p_log = st.text_input("Mot de passe", type="password")
                if st.form_submit_button("Se connecter", use_container_width=True):
                    user = db.verify_user(u_log, p_log)
                    if user:
                        st.session_state.user = user
                        st.rerun()
                    else:
                        st.error("Identifiants incorrects")
        
        # Onglet Inscription
        with mode[1]:
            with st.form("register_form"):
                u_reg = st.text_input("Choisir un pseudo")
                p_reg = st.text_input("Choisir un mot de passe", type="password")
                if st.form_submit_button("S'inscrire", use_container_width=True):
                    if len(u_reg) > 2 and len(p_reg) > 4:
                        new_id = db.create_user(u_reg, p_reg)
                        if new_id:
                            st.success("Compte créé ! Connectez-vous.")
                        else:
                            st.error("Ce nom d'utilisateur est déjà pris.")
                    else:
                        st.warning("Pseudo (>2 car.) ou mot de passe (>4 car.) trop court.")

# Vérification du verrouillage
if st.session_state.user is None:
    show_auth_page()
    st.stop()

# --- 4. VARIABLES DE SESSION UTILISATEUR ---
user_id = st.session_state.user["id"]
username = st.session_state.user["username"]
app_state = st.session_state.app_state

# Mettre à jour user_id si pas encore fait (ex: après connexion)
if app_state.user_id is None:
    app_state.user_id = user_id

def parse_response(content):
    think_match = re.search(r'<think>(.*?)</think>', content, re.DOTALL)
    if think_match:
        thinking = think_match.group(1).strip()
        answer = content.replace(think_match.group(0), "").strip()
        return thinking, answer
    return None, content

# --- 5. SIDEBAR (Control Center) ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2103/2103633.png", width=60)
    st.subheader(f"👤 {username}")
    
    if st.button("🚪 Déconnexion", use_container_width=True):
        st.session_state.user = None
        st.session_state.messages = []
        st.rerun()

    st.markdown("---")
    
    # Récupération des agents propres à l'utilisateur
    user_agents = db.get_user_agents(user_id)
    
    if user_agents:
        agent_names = [a['name'] for a in user_agents]
        selected_name = st.selectbox("#### Agent Actif", agent_names)
        current_agent = next(a for a in user_agents if a['name'] == selected_name)
        
        # Charger l'agent s'il change
        if app_state.current_agent_id != current_agent['id']:
            app_state.load_agent(current_agent['id'])
        
        st.info(f"**Moteur:** {current_agent['vector_method'].upper()}")
    else:
        st.warning("Aucun agent détecté")

    st.markdown("---")



st.markdown("### Knowledge Base")
uploaded_file = st.file_uploader("Nourrir l'intelligence", type=["pdf", "txt"], label_visibility="collapsed")

if uploaded_file and user_agents:
    if st.button("Indexer", use_container_width=True):
        if app_state.current_agent_id is None:
            st.error("Veuillez d'abord sélectionner un agent dans la barre latérale.")
        else:
            with st.spinner("Analyse vectorielle en cours..."):
                # 1. On récupère la méthode de l'agent actif
                # C'est l'étape qui manquait pour éviter l'erreur
                method = app_state.orchestrator.agent_config.get('vector_method', 'tfidf')
                
                # 2. On traite le fichier (en mémoire, pas de fichier temp)
                processor = DocumentProcessor()
                chunks = processor.process_uploaded_file(uploaded_file)
                
                # 3. On stocke selon la méthode
                if method == "sbert":
                    if app_state.vector_db:
                        app_state.vector_db.add_texts(chunks)
                else:
                    if app_state.memory_engine:
                        app_state.memory_engine.documents.extend(chunks)
                        
                        # 4. On recalcule l'indexation (le catalogue)
                        if method == "tfidf":
                            app_state.memory_engine.fit_tfidf()
                        elif method == "cbow":
                            app_state.memory_engine.fit_cbow()
                
                st.toast(f"'{uploaded_file.name}' a été assimilé par l'agent !", icon="✅")
                time.sleep(1)
                st.rerun()
                
# --- 6. DASHBOARD PRINCIPAL ---
tab_chat, tab_manage = st.tabs(["💬 Terminal de Chat", "🛠️ Studio de Création"])

# ONGLET 1 : CHAT
with tab_chat:
    chat_container = st.container(height=250, border=True)
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                thinking, answer = parse_response(message["content"])
                if thinking and message["role"] == "assistant":
                    with st.expander("💭 Réflexion", expanded=False):
                        st.write(thinking)
                st.markdown(answer)

    if prompt := st.chat_input("Posez une question à votre agent..."):
        if not user_agents:
            st.error("Créez d'abord un agent dans le Studio.")
        else:
            st.session_state.messages.append({"role": "user", "content": prompt})
            with chat_container:
                with st.chat_message("user"):
                    st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("L'agent réfléchit..."):
                    response, context = app_state.ask_question(prompt)
                    thinking, answer = parse_response(response)
                    
                    if thinking:
                        with st.expander("💭 Réflexion", expanded=True):
                            st.write(thinking)
                    st.markdown(answer)
                    
                    if context and len(context) > 20:
                        with st.expander("🔍 Sources consultées"):
                            st.write(context)
                    
                    st.session_state.messages.append({"role": "assistant", "content": response})

# ONGLET 2 : STUDIO
with tab_manage:
    st.header("Agent Studio")
    
    with st.expander("Créer un nouvel Agent", expanded=True):
        with st.form("new_agent_form", border=False):
            c1, c2 = st.columns(2)
            with c1:
                name = st.text_input("Nom de l'agent", placeholder="Ex: Jarvis")
                model = st.selectbox("Modèle LLM", ["llama3.1:8b", "mistral-nemo:12b", "gemma3:12b", "phi4:14b"])
                method = st.segmented_control("Méthode RAG", ["tfidf", "sbert", "cbow"], default="sbert")
            with c2:
                temp = st.slider("Température", 0.0, 1.0, 0.4)
                system_prompt = st.text_area("Instructions (Prompt Système)")
            
            st.markdown("---")
            initial_docs = st.file_uploader("Documents initiaux", type=["pdf", "txt"], accept_multiple_files=True)
        
            # Correction : Le bouton de soumission est bien dans le formulaire
            if st.form_submit_button("🔨 Forger l'Agent", use_container_width=True):
                if name and system_prompt:
                    # Enregistrement en DB avec le USER_ID
                    new_agent_id = db.create_agent(user_id, name, model, method, system_prompt, temp)
                    app_state.load_agent(new_agent_id)

                    if initial_docs:
                        processor = DocumentProcessor()
                        for f in initial_docs:
                            chunks = processor.process_uploaded_file(f)
                            # On ajoute directement à la mémoire vectorielle active
                            if method == "sbert":
                                app_state.vector_db.add_texts(chunks)
                            else:
                                app_state.memory_engine.documents.extend(chunks)
                        
                        # Entraîner le vectoriseur après avoir tout chargé
                        if method == "tfidf" and app_state.memory_engine.documents:
                            app_state.memory_engine.fit_tfidf()
                        elif method == "cbow" and app_state.memory_engine.documents:
                            app_state.memory_engine.fit_cbow()

                    st.balloons()
                    st.success(f"Agent {name} déployé avec succès !")
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error("Le nom et les instructions sont obligatoires.")

    st.markdown("---")
    st.subheader("🚢 Ma Flotte")
    if user_agents:
        for agent in user_agents:
            cols = st.columns([3, 2, 2, 1])
            cols[0].write(f"**{agent['name']}**")
            cols[1].caption(f"🤖 {agent['model']}")
            cols[2].caption(f"🧠 {agent['vector_method']}")
            if cols[3].button("🗑️", key=f"del_{agent['id']}"):
                # Suppression sécurisée (agent_id + user_id)
                if db.delete_agent(agent['id'], user_id):
                    # Nettoyage des dossiers si nécessaire
                    path = os.path.join("db_vectors", f"agent_{agent['id']}")
                    if os.path.exists(path):
                        shutil.rmtree(path)
                    st.toast(f"Agent {agent['name']} supprimé.")
                    st.rerun()
    else:
        st.info("Vous n'avez pas encore d'agent.")