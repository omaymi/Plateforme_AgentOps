import streamlit as st
import os
from state_manager import SessionState
from database.database_manager import DatabaseManager

# --- 1. CONFIGURATION & STYLE ---
st.set_page_config(page_title="AgentOps Factory", page_icon="⚡", layout="wide")
st.markdown("""
    <style>
    /* 1. Centrer la ligne des onglets */
    button[data-baseweb="tab"] {
        flex-grow: 1; /* Force les onglets à prendre toute la largeur */
        justify-content: center;
    }

    /* 2. Agrandir le texte et l'espacement des onglets */
    div[data-baseweb="tab-list"] {
        gap: 20px; /* Espace entre les onglets */
        width: 100%;
        display: flex;
        justify-content: center;
    }

    /* Style du texte à l'intérieur des onglets */
    button[data-baseweb="tab"] div p {
        font-size: 22px !important; /* Taille de la police augmentée */
        font-weight: 600 !important;
        padding: 10px 20px;
    }

    /* Barre de sélection sous l'onglet actif */
    div[data-baseweb="tab-highlight"] {
        background-color: #238636 !important; /* Couleur verte pour matcher le bouton */
        height: 4px !important;
    }
    </style>
    """, unsafe_allow_html=True)
# Injection de CSS Custom pour un look "Dark Mode Modern"
st.markdown("""
    <style>
    /* Gradient de fond et police */
    .stApp {
        background: linear-gradient(160deg, #0e1117 0%, #161b22 100%);
    }
    
    /* Style des cartes pour les onglets */
    div[data-baseweb="tab-panel"] {
        background-color: #1c2128;
        padding: 2rem;
        border-radius: 15px;
        border: 1px solid #30363d;
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
    }

    /* Personnalisation de la sidebar */
    section[data-testid="stSidebar"] {
        background-color: #0d1117;
        border-right: 1px solid #30363d;
    }

    /* Boutons stylisés */
    .stButton>button {
        border-radius: 8px;
        transition: all 0.3s ease;
        background-color: #238636;
        color: white;
        border: none;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(35, 134, 54, 0.4);
    }
    </style>
    """, unsafe_allow_html=True)

# Initialisation
if "app_state" not in st.session_state:
    st.session_state.app_state = SessionState()
    st.session_state.messages = []

db = DatabaseManager()
app_state = st.session_state.app_state

# --- 2. FONCTIONS UTILITAIRES ---
def parse_response(content):
    """Extrait le contenu entre les balises <think> et le reste de la réponse."""
    import re
    think_match = re.search(r'<think>(.*?)</think>', content, re.DOTALL)
    if think_match:
        thinking = think_match.group(1).strip()
        answer = content.replace(think_match.group(0), "").strip()
        return thinking, answer
    return None, content

# --- 3. SIDEBAR (Le "Control Center") ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2103/2103633.png", width=90) # Logo fictif
    st.title("AgentOps **Factory**")
    
    st.markdown("---")
    
    # Sélecteur d'agent avec design épuré
    agents = db.get_all_agents()
    if agents:
        agent_names = [a['name'] for a in agents]
        selected_agent_name = st.selectbox("Agent Actif", agent_names)
        agent_id = next(a['id'] for a in agents if a['name'] == selected_agent_name)
        app_state.load_agent(agent_id)
        
        # Badge de statut
        st.info(f"**Mode:** {app_state.orchestrator.agent_config['vector_method'].upper()}")
    else:
        st.warning("⚠️ Aucun agent détecté")

    st.markdown("---")
    
    # Upload avec zone de drop visuelle
    st.markdown("### Knowledge Base")
    uploaded_file = st.file_uploader("Nourrir l'intelligence", type=["pdf", "txt"], label_visibility="collapsed")

    if uploaded_file:
        # On utilise un dossier temporaire avant que state_manager ne le déplace dans le dossier de l'agent
        temp_dir = "temp_files"
        os.makedirs(temp_dir, exist_ok=True)
        file_path = os.path.join(temp_dir, uploaded_file.name)
        
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        if st.button("🚀 Indexer le savoir", use_container_width=True):
            if agents:
                with st.spinner("Vecteurs en cours de création..."):
                    app_state.process_document(file_path)
                    st.toast("Connaissance synchronisée !", icon="✅")
            else:
                st.error("Créez un agent avant d'ajouter des documents.")

# --- 3. DASHBOARD PRINCIPAL ---
tab_chat, tab_manage = st.tabs(["Terminal de Chat", "Studio de Création"])

# --- ONGLET 1 : CHAT (Style Messagerie) ---
with tab_chat:
    # Zone de messages
    chat_container = st.container(height=500, border=False) # Augmenté pour visibilité
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                if message["role"] == "assistant":
                    thinking, answer = parse_response(message["content"])
                    if thinking:
                        with st.expander("💭 Réflexion de l'agent", expanded=False):
                            st.write(thinking)
                    st.markdown(answer)
                else:
                    st.markdown(message["content"])

    # Input flottant
    if prompt := st.chat_input("Envoyez une commande..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with chat_container:
            with st.chat_message("user"):
                st.markdown(prompt)

            # Génération de la réponse
        with st.chat_message("assistant"):
            with st.spinner("L'agent réfléchit..."):
                response, context = app_state.ask_question(prompt)
                
                thinking, answer = parse_response(response)
                if thinking:
                    with st.expander("💭 Réflexion de l'agent", expanded=True):
                        st.write(thinking)
                st.markdown(answer)
            
                if context and len(context.strip()) > 50: 
                    with st.expander("🔍 Sources & Mémoire consultées"):
                        st.write(context)
            
                st.session_state.messages.append({"role": "assistant", "content": response})

# --- ONGLET 2 : STUDIO (Grille moderne) ---
with tab_manage:
    st.header("Agent Studio")
    
    with st.expander("Créer un nouvel Agent", expanded=True):
        with st.form("new_agent_form", border=False):
            c1, c2 = st.columns(2)
            with c1:
                name = st.text_input("Nom unique", placeholder="Ex: Jarvis")
                model = st.selectbox("Modèle LLM", ["mistral", "deepseek-r1"], index=0)
                method = st.segmented_control("Vectorisation", ["tfidf", "sbert", "cbow"], default="sbert")
            with c2:
                temp = st.slider("Créativité (Température)", 0.0, 1.0, 0.4)
                system_prompt = st.text_area("Instructions Système", placeholder="Tu es un expert en...")
            
            # NOUVEAU : Upload de connaissance lors de la création
            st.markdown("---")
            initial_docs = st.file_uploader("Documents de base (Facultatif)", type=["pdf", "txt"], accept_multiple_files=True)
            
            if st.form_submit_button(" Forger l'Agent", use_container_width=True):
                if name and system_prompt:
                    # 1. Création de l'agent en base
                    agent_id = db.create_agent(name, model, method, system_prompt, temp)
                    
                    # 2. Gestion des documents initiaux
                    if initial_docs:
                        # On définit le dossier de destination isolé
                        agent_knowledge_dir = os.path.join("knowledge", f"agent_{agent_id}")
                        os.makedirs(agent_knowledge_dir, exist_ok=True)
                        
                        for uploaded_file in initial_docs:
                            dest_path = os.path.join(agent_knowledge_dir, uploaded_file.name)
                            with open(dest_path, "wb") as f:
                                f.write(uploaded_file.getbuffer())
                    
                    st.balloons()
                    st.success(f"Agent {name} déployé avec sa connaissance !")
                    st.rerun()

    st.markdown("---")
    st.subheader(" Flotte d'Agents")
    all_agents_list = db.get_all_agents() 
    
    if all_agents_list:
        # On affiche une liste avec des boutons de suppression au lieu d'un dataframe statique
        for agent in all_agents_list:
            cols = st.columns([3, 2, 2, 1])
            with cols[0]:
                st.write(f"**{agent['name']}** ({agent['model']})")
            with cols[1]:
                st.caption(f"Vector: {agent['vector_method']}")
            with cols[2]:
                st.caption(f"Temp: {agent['temperature']}")
            with cols[3]:
                if st.button("🗑️", key=f"delete_{agent['id']}", help="Supprimer cet agent"):
                    # 1. Décharger l'agent s'il est actif pour libérer les fichiers (LOCK)
                    if app_state.current_agent_id == agent['id']:
                        app_state.unload_current_agent()
                        st.session_state.messages = [] # On vide aussi l'affichage
                    
                    # 2. Suppression Base de données
                    db.delete_agent(agent['id'])
                    
                    # 3. Nettoyage Fichiers (avec gestion robuste des erreurs Windows)
                    import shutil
                    import time
                    
                    dirs_to_clean = [
                        os.path.join("knowledge", f"agent_{agent['id']}"),
                        os.path.join("db_vectors", f"agent_{agent['id']}"),
                        os.path.join("db_history", f"agent_{agent['id']}"),
                        # Chemins "legacy" (anciennes versions du code)
                        f"db_history_agent_{agent['id']}",
                        f"db_vectors_agent_{agent['id']}"
                    ]
                    
                    # On attend un tout petit peu que les handles se libèrent
                    time.sleep(0.5)
                    
                    for d in dirs_to_clean:
                        if os.path.exists(d):
                            try:
                                shutil.rmtree(d)
                            except PermissionError:
                                # Si Windows bloque encore, on attend encore un peu et on réessaie une fois
                                time.sleep(1)
                                try:
                                    shutil.rmtree(d)
                                except Exception as e:
                                    st.error(f"Erreur lors du nettoyage des fichiers : {e}")
                    
                    st.toast(f"Agent {agent['name']} supprimé.", icon="🗑️")
                    time.sleep(0.5)
                    st.rerun()
    else:
        st.info("Aucun agent configuré pour le moment.")
