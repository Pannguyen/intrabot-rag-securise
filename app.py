import os
import streamlit as st
from dotenv import load_dotenv
import pdfplumber
import numpy as np
from mistralai import Mistral
#from langchain_community.vectorstores import Chroma
from langchain.embeddings.base import Embeddings
from langchain_chroma import Chroma


load_dotenv()
api_key = os.getenv("MISTRAL_API_KEY")

CHROMA_DIR = "./db_chroma"

# Initialiser le client Mistral
client = Mistral(api_key=api_key)

# Chargement texte le PDF
#pdf_path = "/Users/AnhNguyen/Downloads/Politique-RH.pdf"
# Mapping des profils vers leurs documents
pdf_path_profil = {
    "RH": [
        "/Users/AnhNguyen/Downloads/Politique-RH.pdf",
    ],
    "Employé": [
        "/Users/AnhNguyen/Downloads/guide-rag-interactif.pdf",
        "/Users/AnhNguyen/Downloads/guide-integrer-un-salarie.pdf"
    ],
    "Manager": [
        "/Users/AnhNguyen/Downloads/manager.pdf",
        "/Users/AnhNguyen/Downloads/Guide du manager.pdf"
    ],
}

## Ajout fonction pour lire pdf
def lire_pdf(pdf_path): 
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"
    return text.strip()            


# Fonction des chunks
def chunker(texte, taille=2048): 
    return [texte[i:i+taille] for i in range(0, len(texte), taille)]

#chunk_size = 2048
#chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
#print(f"{len(chunks)} chunks créés à partir du PDF.")


## Creation embeddings pour chaque chunks de texte. 
def get_text_embedding(input_text):
    response = client.embeddings.create(
        model="mistral-embed",
        inputs=[input_text]
    )
    return response.data[0].embedding

#text_embeddings = np.load("Politique-RH_embeddings.npy", allow_pickle=True)


class MistralEmbeddings(Embeddings):
    def embed_documents(self, texts):
        return [get_text_embedding(t) for t in texts]
    def embed_query(self, text):
        return get_text_embedding(text)

# Créer la base Chroma
vectordb = Chroma(
    collection_name="intrabot_docs",
    persist_directory=CHROMA_DIR,
    embedding_function=MistralEmbeddings()
)
for doc_id, metadata in zip(vectordb.get()["ids"], vectordb.get()["metadatas"]):
    print(doc_id, metadata)


# Ajouter les textes s’il n’y a encore rien


if len(vectordb.get()["ids"]) == 0:
    # Ajouter les textes (toujours reconstruire)
    for profil, fichiers in pdf_path_profil.items():
        for chemin_pdf in fichiers:
            texte = lire_pdf(chemin_pdf)
            chunks = chunker(texte)
            metadatas = [{"profil_autorise": profil, "source": os.path.basename(chemin_pdf)} for _ in chunks]
            emb_path = f"embeddings_{profil}.npy"
            if os.path.exists(emb_path):
                text_embeddings = np.load(emb_path, allow_pickle=True)
            else:
                text_embeddings = np.array([get_text_embedding(c) for c in chunks])
                np.save(emb_path, text_embeddings)
                #print(f"Embeddings sauvegardés dans {emb_path}")

            # Ajouter au vecteurstore
            vectordb.add_texts(texts=chunks, metadatas=metadatas)
            #print(f"{chemin_pdf} indexé ({len(chunks)} chunks) pour le profil {profil}.")

    vectordb.persist()


#     for profil, fichiers in pdf_path_profil.items():
#         for chemin_pdf in fichiers:
#             texte = lire_pdf(chemin_pdf)
#             chunks = chunker(texte)
#             metadatas = [{"profil_autorise": profil, "source": os.path.basename(chemin_pdf)} for _ in chunks]
#             vectordb.add_texts(texts=chunks, metadatas=metadatas)
#             print(f"{chemin_pdf} indexé ({len(chunks)} chunks) pour le profil {profil}.")
#     vectordb.persist()
   # metadatas = [{"profil_autorise": "RH"} for _ in chunks]
    #vectordb.add_texts(texts=chunks, metadatas=metadatas)
    #vectordb.persist()
    #print(" Base vectorielle Chroma initialisée avec les documents RH.")


# -------------------------------------------------------
#  Fonction de recherche filtrée
# -------------------------------------------------------
def retrieve_docs(query: str, profil: str, k: int = 3):
    results = vectordb.similarity_search_with_score(
        query=query,
        k=k,
        filter={"profil_autorise": profil}
        #filter={"profil_autorise": {"$eq": profil}}
    )
    return results


#Fonction de génération de réponse avec Mistral
# -------------------------------------------------------
def generate_answer(query: str, profil: str) -> str:
    docs = retrieve_docs(query, profil)
    if not docs:
        return "Aucun document accessible pour ce profil."

    # Extraire le contenu textuel des documents retournés
    context = "\n\n".join([doc.page_content for doc, _ in docs])

    prompt = f"""
Pour le profil {profil}.
Contexte :
{context}
Question :
{query}
Réponse :
"""
    # Appel du modèle Mistral pour la génération
    response = client.chat.complete(
        model="mistral-small",  
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content.strip()

# Interface Streamlit
# -------------------------------------------------------
st.set_page_config(page_title="Assistant Mistral", layout="wide")
st.title("Intrabot — Assistant documentaire ")

st.sidebar.header("Paramètres")
profil = st.sidebar.selectbox("Profil utilisateur :", ["RH", "Employé", "Manager"])
st.sidebar.info("Choisis ton profil pour filtrer les documents.")

st.write("Posez une question :")
##query = st.text_area(" Votre question :", placeholder="Ex: Quels sont les congés autorisés ?")
query = st.text_area(" Votre question :") 
if st.button("Générer la réponse"):
        with st.spinner("Recherche dans les documents et génération de la réponse..."):
            answer = generate_answer(query, profil)
        st.success("Réponse générée :")
        st.write(answer)

st.markdown("---")
st.caption("Propulsé par Mistral AI + LangChain + Streamlit")