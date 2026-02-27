from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import PyPDF2
import io

class DocumentProcessor:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100
        )

    def process_uploaded_file(self, uploaded_file):
        file_extension = uploaded_file.name.split(".")[-1].lower()

        # 1️⃣ Extraction texte
        if file_extension == "pdf":
            pdf_reader = PyPDF2.PdfReader(uploaded_file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() or ""

        elif file_extension == "txt":
            text = uploaded_file.read().decode("utf-8")

        else:
            raise ValueError("Format non supporté. Utilisez .pdf ou .txt")

        # 2️⃣ Convertir en Document LangChain
        doc = Document(page_content=text)

        # 3️⃣ Chunking
        chunks = self.text_splitter.split_documents([doc])

        # 4️⃣ Retourner uniquement le texte
        return [chunk.page_content for chunk in chunks]