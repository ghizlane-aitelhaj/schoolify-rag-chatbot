import os
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# Loaders LangChain
from langchain_community.document_loaders import (
    PyMuPDFLoader,                      # PDF
    TextLoader,                         # TXT
    UnstructuredMarkdownLoader,         # .md
    UnstructuredWordDocumentLoader,     # .docx
    CSVLoader                           # .csv
)

# OCR pour les images
from PIL import Image
import pytesseract
from langchain.schema.document import Document  

# Chargement du .env
load_dotenv()

api_key_check = os.getenv("OPENAI_API_KEY")
if api_key_check:
    print(f"✅ Clé API OpenAI trouvée : {api_key_check[:5]}*****")
else:
    print("❌ Clé API OpenAI non trouvée. Vérifie ton fichier .env.")

# Dossier des documents
DOCS_DIR = "documents"
if not os.path.exists(DOCS_DIR):
    os.makedirs(DOCS_DIR)

def ocr_image_to_document(file_path: str) -> list:
    """Effectue l’OCR sur une image et retourne une liste [Document]"""
    try:
        img = Image.open(file_path)
        text = pytesseract.image_to_string(img)
        if not text.strip():
            print(f"❗ Aucun texte détecté dans l'image : {file_path}")
            return []
        print(f"🧾 Texte extrait de l'image ({len(text)} caractères)")
        doc = Document(page_content=text, metadata={"source": file_path})
        return [doc]
    except Exception as e:
        print(f"❌ Erreur OCR sur l'image {file_path} : {e}")
        return []

def process_document(file_path: str):
    print(f"📄 Traitement du fichier : {file_path}")

    # Sélection du bon loader selon l’extension
    try:
        if file_path.endswith(".pdf"):
            loader = PyMuPDFLoader(file_path)
            documents = loader.load()
        elif file_path.endswith(".txt"):
            loader = TextLoader(file_path)
            documents = loader.load()
        elif file_path.endswith(".md"):
            loader = UnstructuredMarkdownLoader(file_path)
            documents = loader.load()
        elif file_path.endswith(".docx"):
            loader = UnstructuredWordDocumentLoader(file_path)
            documents = loader.load()
        elif file_path.endswith(".csv"):
            loader = CSVLoader(file_path)
            documents = loader.load()
        elif file_path.endswith((".png", ".jpg", ".jpeg")):
            documents = ocr_image_to_document(file_path)
        else:
            print(f"⚠️ Format non supporté : {file_path}")
            return None

        if not documents:
            print("❌ Aucun contenu extrait.")
            return None

        print(f"📚 {len(documents)} document(s) chargé(s)")

        # chunking
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=300
        )
        chunks = splitter.split_documents(documents)
        print(f"✂️ {len(chunks)} chunks générés.")

        # Embedding
        embeddings = OpenAIEmbeddings()

        # FAISS
        if os.path.exists("faiss_index"):
            db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
            db.add_documents(chunks)
            print("✅ Ajout aux documents existants dans FAISS.")
        else:
            db = FAISS.from_documents(chunks, embeddings)
            print("✅ Nouvelle base FAISS créée.")

        db.save_local("faiss_index")
        print("💾 Base FAISS sauvegardée.")
        return db

    except Exception as e:
        print(f"❌ Erreur lors du traitement : {e}")
        return None

def get_vector_store():
    embeddings = OpenAIEmbeddings()
    if os.path.exists("faiss_index"):
        print("📦 Chargement de la base FAISS existante...")
        return FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    print("⚠️ Aucune base FAISS trouvée.")
    return None

# bach ykoun exportable dans app.py
DOCS_DIR = DOCS_DIR
