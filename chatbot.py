import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import FAISS
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from data_processor import get_vector_store

load_dotenv()

system_instruction_content = """
Tu es un **assistant pédagogique **. Ton objectif est d'aider les étudiants à comprendre les concepts complexes de manière simple et claire.

**Voici les règles pour tes réponses :**
- **Clarté et simplicité :** Utilise un langage facile à comprendre, évite le jargon technique inutile.
- **Structure :** Organise tes réponses avec des titres (##), des sous-titres (###), des listes à puces (-), et du texte en gras (**) pour faciliter la lecture.
- **Exemples Pratiques :** Si possible, illustre les concepts avec des exemples concrets ou des scénarios d'application.
- **Définition Complète :** Pour une question sur un concept (ex: APIPA, IPv4, DHCP), donne toujours :
  1. **Une définition claire.**
  2. **Son utilité ou son objectif.**
  3. **Son fonctionnement (étapes clés si applicable).**
  4. **Des exemples ou applications pratiques.**
- **Concision :** Sois direct et va droit au but, mais assure-toi de couvrir les informations essentielles.
- **Pertinence :** Réponds uniquement à la question posée. Si l'information n'est pas disponible, indique-le poliment.
- **Mise en forme Markdown :** Utilise le Markdown pour des réponses bien formatées (listes, gras, titres).

**Si une image est fournie :**
- Analyse l'image attentivement et réponds à la question de l'utilisateur en te basant sur le contenu de l'image.
- Intègre les informations de l'image dans ta réponse de manière fluide.
- Mentionne si ta réponse est basée sur l'image fournie.

N'hésite pas à me poser d'autres questions !
"""

# Prompt principal pour LLM 
fallback_prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(system_instruction_content),
    MessagesPlaceholder(variable_name="chat_history"),
    HumanMessagePromptTemplate.from_template("{question}")
])

# Prompt pour la chaîne qui combine les documents
combine_docs_prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(system_instruction_content),
    HumanMessagePromptTemplate.from_template(
        "Voici les documents extraits du contexte :\n\n{context}\n\nEn te basant sur ces documents, réponds à la question suivante :\n{question}"
    )
])

llm = ChatOpenAI(model_name="gpt-4o", temperature=0.5)

def custom_qa_chain(question: str, chat_history: list):
    vector_store = get_vector_store()

    formatted_history = []
    for msg in chat_history:
        if msg["role"] == "user":
            formatted_history.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            formatted_history.append(AIMessage(content=msg["content"]))

    if not vector_store:
        response = llm.predict_messages(
            fallback_prompt.format_prompt(chat_history=formatted_history, question=question).to_messages()
        )
        return {"answer": response.content, "source_documents": []}

    retriever = vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"score_threshold": 0.7, "k": 3}
    )

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key="answer")
    for msg in chat_history:
        if msg["role"] == "user":
            memory.chat_memory.add_user_message(msg["content"])
        elif msg["role"] == "assistant":
            memory.chat_memory.add_ai_message(msg["content"])

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": combine_docs_prompt},  
        return_source_documents=True,
        output_key="answer"
    )

    result = qa_chain.invoke({"question": question, "chat_history": chat_history})
    return result

def create_chatbot_chain():
    vector_store = get_vector_store()
    if not vector_store:
        print("Vector database non trouvée.")
        return None

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key="answer")
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(),
        memory=memory,
        return_source_documents=True,
        output_key="answer"
    )
    return qa_chain
