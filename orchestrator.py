from langchain_ollama import OllamaLLM # Version à jour
from langchain_core.prompts import ChatPromptTemplate
from database.database_manager import DatabaseManager
import config
## Logique de génération LLM (LangChain)

class AgentOrchestrator:
    def __init__(self, agent_id, user_id):
        self.db = DatabaseManager()
        self.agent_config = self.db.get_agent_by_id(agent_id, user_id)
        
        if not self.agent_config:
            raise ValueError("Agent non trouvé dans la base de données.")

        # 1. Configuration du LLM
        self.llm = OllamaLLM(
            model=self.agent_config['model'],
            temperature=self.agent_config['temperature'],
            base_url=config.OLLAMA_HOST,
            timeout=120
        )

        # 2. Le "Moule" de réponse (Prompt Engineering)
        self.template = """
        Tu es l'agent : {name}.
        Instructions : {system_prompt}
        Historique de la conversation :
        {chat_history}
        
        Sers-toi du CONTEXTE suivant pour répondre à la question. 
        Si la réponse n'est pas dans le contexte, dis que tu ne sais pas, n'invente rien.
        
        CONTEXTE :
        {context}
        
        QUESTION :
        {question}
        
        RÉPONSE :
        """
        self.prompt_template = ChatPromptTemplate.from_template(self.template)

    def generate_response(self, question, context, chat_history=""):
        chain = self.prompt_template | self.llm
    
        response = chain.invoke({
            "name": self.agent_config['name'],
            "system_prompt": self.agent_config['system_prompt'],
            "chat_history": chat_history, # La mémoire courte est injectée ici
            "context": context,           # Le RAG est injecté ici
            "question": question
        })
        return response