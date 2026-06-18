import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Carica il modello
model = SentenceTransformer('allenai-specter')

# Carica l'Aims & Scope
with open("data/aims_and_scope.txt", "r") as f:
    aims_and_scope = f.read()

# Carica gli abstract
with open("data/abstracts.json", "r") as f:
    articoli = json.load(f)

# Embedding dell'Aims & Scope
print("Generando embedding Aims & Scope...")
embedding_aims = model.encode(aims_and_scope)

# Embedding degli abstract
print("Generando embedding abstract...")
abstracts_testi = [articolo["abstract"] for articolo in articoli]
embeddings_abstract = model.encode(abstracts_testi, show_progress_bar=True)

# Salva gli embedding
np.save("data/embedding_aims.npy", embedding_aims)
np.save("data/embeddings_abstract.npy", embeddings_abstract)

print("Embedding salvati!")

from sklearn.metrics.pairwise import cosine_similarity

# Calcola alignment scores
scores = cosine_similarity([embedding_aims], embeddings_abstract)[0]

# Aggiungi gli scores agli articoli
for i, articolo in enumerate(articoli):
    articolo["alignment_score"] = float(scores[i])

# Salva i risultati
with open("results/results.json", "w") as f:
    json.dump(articoli, f, indent=2)

print("Alignment scores salvati in results/results.json")