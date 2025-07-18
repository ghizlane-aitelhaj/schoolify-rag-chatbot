import streamlit as st
import os
import base64
import shutil

try:
    from data_processor import process_document, get_vector_store, DOCS_DIR
    from chatbot import create_chatbot_chain, custom_qa_chain
except ImportError:
    st.error("Les fichiers 'data_processor.py' ou 'chatbot.py' sont manquants ou contiennent des erreurs.")
    st.info("Veuillez vous assurer qu'ils sont présents dans le même répertoire que votre script principal.")
    def process_document(path): st.info(f"Traitement fictif de {path}")
    def get_vector_store(): return None
    DOCS_DIR = "uploaded_docs"
    def create_chatbot_chain(): return None
    def custom_qa_chain(prompt, history): return {"answer": f"Réponse fictive à : {prompt}"}

# Set page configuration
st.set_page_config(
    page_title="Schoolify Chatbot RAG",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS for improved styling
st.markdown("""
    <style>
        /* Animation keyframes */
        @keyframes fadeSlide {
            from { opacity: 0; transform: translateY(-20px); }
            to { opacity: 1; transform: translateY(0); }
        }

        /* Theme variables */
        :root {
            --text-color: #ffffff;
            --chat-user-bg: #2a2a2a;
            --chat-assistant-bg: #003366;
            --banner-gradient-start: #0f2027;
            --banner-gradient-mid: #203a43;
            --banner-gradient-end: #2c5364;
            --banner-sub-color: #ccc;
            --banner-title-color: white;
            --button-bg: #0a9396;
            --button-hover-bg: #087b7c;
            --button-text-color: white;
            --sidebar-header-color: #0a9396;
            --shadow-color: rgba(0, 0, 0, 0.2);
            --primary-background-color: #1a1a1a;
        }

        body[data-theme="light"] {
            --text-color: #333333;
            --chat-user-bg: #e6f3ff;
            --chat-assistant-bg: #d9e6ff;
            --banner-gradient-start: #a8dadc;
            --banner-gradient-mid: #457b9d;
            --banner-gradient-end: #1d3557;
            --banner-sub-color: #333333;
            --banner-title-color: white;
            --button-bg: #1d3557;
            --button-hover-bg: #2a4a7b;
            --button-text-color: white;
            --sidebar-header-color: #1d3557;
            --shadow-color: rgba(0, 0, 0, 0.1);
            --primary-background-color: #f0f2f5;
        }

        /* Global styling */
        html, body {
            background-color: var(--primary-background-color);
            color: var(--text-color);
            transition: background-color 0.3s, color 0.3s;
        }

        /* Banner styling */
        .banner-container {
            background: linear-gradient(135deg, var(--banner-gradient-start), var(--banner-gradient-mid), var(--banner-gradient-end));
            padding: 2rem;
            border-radius: 1.2rem;
            margin-bottom: 2rem;
            text-align: center;
            box-shadow: 0 8px 20px var(--shadow-color);
            animation: fadeSlide 0.5s ease-out;
        }

        .banner-title {
            color: var(--banner-title-color);
            font-size: 2.2rem;
            margin: 0;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 10px;
            flex-wrap: wrap;
        }

        .title-inline-logo {
            height: 1.8em;
        }

        .banner-sub {
            color: var(--banner-sub-color);
            font-size: 1.1rem;
            margin-top: 0.5rem;
        }

        /* Chat message styling */
        .chat-message {
            padding: 1rem;
            margin-bottom: 1rem;
            border-radius: 0.8rem;
            box-shadow: 0 3px 8px var(--shadow-color);
            display: flex;
            align-items: center;
            transition: transform 0.2s ease, box-shadow 0.2s ease;
            color: var(--text-color);
        }
        .chat-message:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 12px var(--shadow-color);
        }
        .user {
            background-color: var(--chat-user-bg);
            justify-content: flex-start;
            padding: 1rem 1.5rem;
        }
        .assistant {
            background-color: var(--chat-assistant-bg);
            justify-content: flex-start;
            padding: 1rem 1.5rem;
        }

        .chatbot-icon {
            width: 32px;
            height: 32px;
            border-radius: 50%;
            margin-right: 12px;
            object-fit: contain;
        }
        .user-icon {
            font-size: 1.8em;
            margin-right: 12px;
            color: var(--button-bg);
        }
        .user .user-icon {
            order: -1;
        }

        /* Sidebar styling */
        section[data-testid="stSidebar"] {
            background-color: var(--primary-background-color);
            padding: 1rem;
            border-right: 1px solid var(--shadow-color);
            color: var(--text-color);
        }

        /* Button styling */
        .stButton>button {
            background-color: var(--button-bg);
            color: var(--button-text-color);
            border-radius: 0.6rem;
            padding: 0.6rem 1.2rem;
            margin-top: 1rem;
            width: 100%;
            font-weight: 500;
            transition: background-color 0.3s, transform 0.2s;
        }
        .stButton>button:hover {
            background-color: var(--button-hover-bg);
            transform: translateY(-2px);
        }

        /* File uploader styling */
        div[data-testid="stFileUploader"] {
            border: 1px dashed var(--button-bg);
            border-radius: 0.6rem;
            padding: 1rem;
            color: var(--text-color);
        }
        div[data-testid="stFileUploader"] > label > div > p {
            color: var(--text-color);
            font-size: 0.9rem;
        }

        /* Responsive design */
        @media (max-width: 768px) {
            .banner-title {
                font-size: 1.8rem;
            }
            .banner-sub {
                font-size: 0.9rem;
            }
            .chat-message {
                padding: 0.8rem;
            }
        }
    </style>
""", unsafe_allow_html=True)

# Load logo
logo_path = os.path.join("images", "logo_schoolify-removebg-preview.png")
if not os.path.exists(logo_path):
    st.error(f"Erreur : Le fichier logo n'a pas été trouvé à l'emplacement : {logo_path}")
    st.stop()

with open(logo_path, "rb") as f:
    logo_base64 = base64.b64encode(f.read()).decode()

# Banner
with st.container():
    st.markdown(
        f"""
        <div class="banner-container" role="banner">
            <h1 class="banner-title" aria-label="Schoolify Chatbot RAG">
                Schoolify Chatbot RAG
                <img class="title-inline-logo" src="data:image/png;base64,{logo_base64}" alt="Schoolify Logo">
            </h1>
            <p class="banner-sub">📚 Ton assistant pédagogique intelligent</p>
        </div>
        """,
        unsafe_allow_html=True
    )

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'qa_chain' not in st.session_state:
    st.session_state.qa_chain = create_chatbot_chain() if os.path.exists("faiss_index") else None

# Sidebar
with st.sidebar:
    st.header("📂 Gestion des documents", anchor=False)
    st.info("Changez le thème clair/sombre via les paramètres Streamlit (⚙️ en haut à droite).")

    with st.container():
        uploaded_files = st.file_uploader(
            "Déposez vos fichiers ici",
            type=["pdf", "txt", "md", "docx", "csv", "png", "jpg", "jpeg"],
            accept_multiple_files=True,
            help="Formats supportés : PDF, TXT, MD, DOCX, CSV, PNG, JPG, JPEG"
        )

        if uploaded_files and st.button("🚀 Traiter les documents", key="process_docs"):
            with st.spinner("Traitement en cours..."):
                os.makedirs(DOCS_DIR, exist_ok=True)
                for uploaded_file in uploaded_files:
                    path = os.path.join(DOCS_DIR, uploaded_file.name)
                    with open(path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    process_document(path)
                st.success("✅ Documents traités avec succès.")
                st.session_state.qa_chain = create_chatbot_chain()
                st.rerun()

        if st.button("🧹 Réinitialiser documents", key="reset_docs"):
            if os.path.exists(DOCS_DIR):
                for file in os.listdir(DOCS_DIR):
                    os.remove(os.path.join(DOCS_DIR, file))
                if not os.listdir(DOCS_DIR):
                    os.rmdir(DOCS_DIR)
            if os.path.exists("faiss_index"):
                shutil.rmtree("faiss_index")
            st.session_state.qa_chain = None
            st.session_state.chat_history = []
            st.success("✅ Documents et historique réinitialisés.")
            st.rerun()

        if st.button("🆕 Nouveau Chat", key="new_chat"):
            st.session_state.chat_history = []
            st.success("✅ Nouvelle conversation démarrée.")
            st.rerun()

    st.markdown("---")
    st.subheader("📄 Documents chargés", anchor=False)
    docs = os.listdir(DOCS_DIR) if os.path.exists(DOCS_DIR) else []
    if docs:
        for d in docs:
            st.markdown(f"✅ {d}")
    else:
        st.markdown("_Aucun document actuellement._")

# Chat history display
with st.container():
    for msg in st.session_state.chat_history:
        role_class = "user" if msg["role"] == "user" else "assistant"
        content = msg["content"].replace("<", "<").replace(">", ">")  
        if msg["role"] == "user":
            st.markdown(
                f'<div class="chat-message {role_class}" role="log" aria-label="Message utilisateur"><span class="user-icon">👤</span><span>{content}</span></div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f'<div class="chat-message {role_class}" role="log" aria-label="Réponse du chatbot"><img src="data:image/png;base64,{logo_base64}" class="chatbot-icon" alt="Chatbot Icon"><span>{content}</span></div>',
                unsafe_allow_html=True
            )

# Chat input
if prompt := st.chat_input("Pose ta question ici...", key="chat_input"):
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    st.markdown(
        f'<div class="chat-message user" role="log" aria-label="Message utilisateur"><span class="user-icon">👤</span><span>{prompt.replace("<", "<").replace(">", ">")}</span></div>',
        unsafe_allow_html=True
    )

    if st.session_state.qa_chain:
        with st.spinner("🤖 Réflexion en cours..."):
            try:
                result = custom_qa_chain(prompt, st.session_state.chat_history)
                answer = result["answer"]
                print(f"Raw answer: {answer}")
                answer = answer.replace("# ", "") if answer.startswith("# ") else answer
                if prompt.lower() in answer.lower():
                    answer = answer.replace(prompt, "").strip()
                st.session_state.chat_history.append({"role": "assistant", "content": answer})
                st.markdown(
                    f'<div class="chat-message assistant" role="log" aria-label="Réponse du chatbot"><img src="data:image/png;base64,{logo_base64}" class="chatbot-icon" alt="Chatbot Icon"><span>{answer.replace("<", "<").replace(">", ">")}</span></div>',
                    unsafe_allow_html=True
                )

                source_docs = result.get("source_documents", [])
                if source_docs:
                    with st.sidebar:
                        st.markdown("---")
                        st.subheader("📚 Sources utilisées", anchor=False)
                        for i, doc in enumerate(source_docs):
                            src = doc.metadata.get('source', 'Document inconnu')
                            page = doc.metadata.get('page', 'N/A')
                            st.markdown(f"**{i+1}. {os.path.basename(src)}** (Page: {page})")
                            with st.expander(f"Extrait document {i+1}"):
                                excerpt = doc.page_content[:500]
                                st.markdown(f"{excerpt}...")
            except Exception as e:
                error_message = "❌ Erreur lors du traitement. Veuillez réessayer ou charger d'autres documents."
                st.session_state.chat_history.append({"role": "assistant", "content": error_message})
                st.markdown(
                    f'<div class="chat-message assistant" role="log" aria-label="Erreur du chatbot"><img src="data:image/png;base64,{logo_base64}" class="chatbot-icon" alt="Chatbot Icon"><span>{error_message}</span></div>',
                    unsafe_allow_html=True
                )
                st.error(f"Une erreur est survenue : {e}")
    else:
        warning_message = "📄 Veuillez d'abord charger et traiter des documents pour démarrer la conversation."
        st.session_state.chat_history.append({"role": "assistant", "content": warning_message})
        st.markdown(
            f'<div class="chat-message assistant" role="log" aria-label="Avertissement du chatbot"><img src="data:image/png;base64,{logo_base64}" class="chatbot-icon" alt="Chatbot Icon"><span>{warning_message}</span></div>',
            unsafe_allow_html=True
        )