
from mistralai import Mistral
import numpy as np
import PyPDF2
import os
from dotenv import load_dotenv


#api_key = "x62WtEYyJ4rKaqDQSWKelQUtg4mvNcNi"
load_dotenv()
api_key = os.getenv("MISTRAL_API_KEY")

client = Mistral(api_key=api_key)
response = client.chat.complete(
    model="mistral-small",
    messages=[{"role": "user", "content": "Qui est le président de la France?."}]
)

print(response)


# Lire le PDF
pdf_path = "/Users/AnhNguyen/Downloads/Politique-RH.pdf"
text = ""
with open(pdf_path, "rb") as file:
    reader = PyPDF2.PdfReader(file)
    for page in reader.pages:
        text += page.extract_text() + "\n"

print(f"PDF chargé ({len(text)} caractères)")

# Découper le texte en morceaux
chunk_size = 2048
chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
print(f"{len(chunks)} chunks créés")

#Fonction pour obtenir les embeddings
def get_text_embedding(input_text):
    embeddings_response = client.embeddings.create(
        model="mistral-embed",
        inputs=input_text
    )
    return np.array(embeddings_response.data[0].embedding)

# Calcul des embeddings
text_embeddings = np.array([get_text_embedding(chunk) for chunk in chunks])


# Sauvegarder les embeddings pour éviter de les recalculer à chaque fois
np.save("Politique-RH_embeddings.npy", text_embeddings)


