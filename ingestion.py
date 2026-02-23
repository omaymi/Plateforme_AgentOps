from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os

class DocumentProcessor:
    def __init__(self):
        # On définit comment on découpe le texte
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, # 1000 caractères par morceau
            chunk_overlap=100 # On garde un petit lien entre les morceaux
        )

    def process_file(self, file_path):
        file_extension = os.path.splitext(file_path)[1].lower()
        
        # 1. Chargement selon l'extension
        if file_extension == ".pdf":
            loader = PyPDFLoader(file_path)
        elif file_extension == ".txt":
            loader = TextLoader(file_path, encoding="utf-8")
        else:
            raise ValueError("Format non supporté. Utilisez .pdf ou .txt")

        # 2. Découpage en morceaux (Chunks)
        documents = loader.load()
        chunks = self.text_splitter.split_documents(documents)
        
        # On retourne juste le texte de chaque morceau pour ton VectorEngine
        return [chunk.page_content for chunk in chunks]

