import json
import numpy as np
import matplotlib.pyplot as plt

# Carica i risultati
with open("results/results.json", "r") as f:
    articoli = json.load(f)


scores = np.array([a["alignment_score"] for a in articoli])
anni = [a["year"] for a in articoli]
titoli = [a["title"] for a in articoli]

print(f"Articles in the 2019-2025 period: {len(articoli)}")

# Estrai gli scores
scores = np.array([a["alignment_score"] for a in articoli])
anni = [a["year"] for a in articoli]
titoli = [a["title"] for a in articoli]

# Istogramma degli alignment scores
plt.figure(figsize=(10, 6))
plt.hist(scores, bins=30, color='steelblue', edgecolor='black')
plt.axvline(scores.mean(), color='red', linestyle='--', label=f'Mean: {scores.mean():.3f}')
plt.xlabel('Alignment Score')
plt.ylabel('Number of articles')
plt.title('Distribution of Alignment Scores - EMNLP')
plt.legend()
plt.tight_layout()
plt.savefig("results/histogram.png")
plt.show()
print("Histogram saved!")



# Drift temporale
from collections import defaultdict

scores_per_anno = defaultdict(list)
for a in articoli:
    if a["year"]:
        scores_per_anno[a["year"]].append(a["alignment_score"])

anni_ordinati = sorted(scores_per_anno.keys())
medie_annuali = [np.mean(scores_per_anno[a]) for a in anni_ordinati]

plt.figure(figsize=(12, 6))
plt.plot(anni_ordinati, medie_annuali, marker='o', color='steelblue')
plt.xlabel('year')
plt.ylabel('Average Alignment Score')
plt.title('Temporal Drift of Alignment Scores - EMNLP')
plt.grid(True)
plt.tight_layout()
plt.savefig("results/drift_temporale.png")
plt.show()
print("Temporal drift plot saved!")



# Numero di articoli per anno
conteggio_per_anno = [len(scores_per_anno[a]) for a in anni_ordinati]

fig, ax1 = plt.subplots(figsize=(12, 6))

# Linea degli scores medi
ax1.plot(anni_ordinati, medie_annuali, marker='o', color='steelblue', label='Average Score')
ax1.set_xlabel('year')
ax1.set_ylabel('Average Alignment Score', color='steelblue')
ax1.tick_params(axis='y', labelcolor='steelblue')

# Barre del conteggio articoli su asse secondario
ax2 = ax1.twinx()
ax2.bar(anni_ordinati, conteggio_per_anno, alpha=0.3, color='orange', label='Number of articles')
ax2.set_ylabel('Number of articles', color='orange')
ax2.tick_params(axis='y', labelcolor='orange')

plt.title('Temporal Drift and Number of Articles per Year - EMNLP')
fig.legend(loc='upper left', bbox_to_anchor=(0.1, 0.9))
plt.tight_layout()
plt.savefig("results/drift_temporale_contestualizzato.png")
plt.show()
print("Contextualized plot saved!")



# Outlier
soglia_bassa = scores.mean() - 2*scores.std()
soglia_alta = scores.mean() + 2*scores.std()



# Conteggio outlier
outlier_bassi_list = [a for a in articoli if a["alignment_score"] < soglia_bassa]
outlier_alti_list = [a for a in articoli if a["alignment_score"] > soglia_alta]

print(f"\nLower outlier threshold: {soglia_bassa:.4f}")
print(f"Upper outlier threshold: {soglia_alta:.4f}")
print(f"\nNumber of low outliers: {len(outlier_bassi_list)}")
for a in outlier_bassi_list:
    print(f"  Score: {a['alignment_score']:.4f} | year: {a['year']} | {a['title'][:60]}")

print(f"\nNumber of upper outliers: {len(outlier_alti_list)}")
for a in outlier_alti_list:
    print(f"  Score: {a['alignment_score']:.4f} | year: {a['year']} | {a['title'][:60]}")

# Rappresentazione grafica degli outlier
plt.figure(figsize=(12, 6))
plt.scatter(range(len(articoli)), 
            sorted(scores), 
            color='steelblue', alpha=0.5, s=20, label='Normal articles')
plt.axhline(soglia_bassa, color='red', linestyle='--', label=f'Lower threshold: {soglia_bassa:.3f}')
plt.axhline(soglia_alta, color='green', linestyle='--', label=f'Upper threshold: {soglia_alta:.3f}')
plt.axhline(scores.mean(), color='orange', linestyle='-', label=f'Mean: {scores.mean():.3f}')

# Evidenzia gli outlier
scores_ordinati = sorted(scores)
outlier_bassi_idx = [i for i, s in enumerate(scores_ordinati) if s < soglia_bassa]
outlier_alti_idx = [i for i, s in enumerate(scores_ordinati) if s > soglia_alta]

plt.scatter(outlier_bassi_idx, 
            [scores_ordinati[i] for i in outlier_bassi_idx], 
            color='red', s=50, zorder=5, label=f'Lower outlier ({len(outlier_bassi_idx)})')
plt.scatter(outlier_alti_idx, 
            [scores_ordinati[i] for i in outlier_alti_idx], 
            color='green', s=50, zorder=5, label=f'Upper outlier ({len(outlier_alti_idx)})')

plt.xlabel('Articles (sorted by score)')
plt.ylabel('Alignment Score')
plt.title('Distribution of Alignment Scores with Outliers - EMNLP')
plt.legend()
plt.tight_layout()
plt.savefig("results/outlier_plot.png")
plt.show()
print("Contextualized outlier plot saved!")