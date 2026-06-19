import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load the model
model = SentenceTransformer('allenai-specter')

# Load the Aims & Scope
with open("data/aims_and_scope.txt", "r") as f:
    aims_and_scope = f.read()

# Load the abstracts
with open("data/abstracts.json", "r") as f:
    articoli = json.load(f)

# Embedding of the Aims & Scope
print("Generating Aims & Scope embeddings...")
embedding_aims = model.encode(aims_and_scope)

# Embedding of the abstract
print("Generating abstract embeddings...")
abstracts_testi = [articolo["abstract"] for articolo in articoli]
embeddings_abstract = model.encode(abstracts_testi, show_progress_bar=True)

# Save the embeddings
np.save("data/embedding_aims.npy", embedding_aims)
np.save("data/embeddings_abstract.npy", embeddings_abstract)

print("Embedding saved!")

from sklearn.metrics.pairwise import cosine_similarity

# Calculate alignment scores
scores = cosine_similarity([embedding_aims], embeddings_abstract)[0]

# Add the scores to the articles
for i, articolo in enumerate(articoli):
    articolo["alignment_score"] = float(scores[i])

# Save the results
with open("results/results.json", "w") as f:
    json.dump(articoli, f, indent=2)

print("Alignment scores saved in results/results.json")