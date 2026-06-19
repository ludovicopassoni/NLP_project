# Thematic Alignment Analysis in EMNLP Conference Papers

This project analyzes the thematic alignment between papers published at EMNLP 
(Empirical Methods in Natural Language Processing) and the conference's stated 
Aims & Scope, using sentence embeddings and cosine similarity.

## Overview

Academic venues define their intended focus through an Aims & Scope statement. 
This project investigates whether papers actually published at EMNLP reflect 
this stated scope, and whether thematic alignment has changed over time 
(2019-2025).

## Methodology

1. **Data collection** — Paper abstracts are collected from Semantic Scholar 
   using its Academic Graph API, filtering by EMNLP venue.
2. **Embedding generation** — Abstracts and the Aims & Scope text are encoded 
   using `allenai-specter`, a BERT-based model pre-trained on scientific literature.
3. **Alignment computation** — An alignment score is computed for each paper as 
   the cosine similarity between its embedding and the Aims & Scope embedding.
4. **Analysis** — Score distribution, temporal drift, and outlier detection.

## Repository Structure
NLP_project/

├── data/

│   ├── abstracts.json              # Collected EMNLP abstracts

│   ├── embedding_aims.npy          # Embedding of the Aim&Scope

│   ├── embeddings_abstract.npy     # Embedding of the Abstracts

│   └── aims_and_scope.txt          # Reference Aims & Scope text

├── notebooks/

│   └── analysis.ipynb          # Full pipeline with explanations and results

├── paper/sn-article-template

│   └── sn-article.tex          # Final paper (Springer format)

├── results/

│   ├── results.json            # Alignment scores per paper

│   └── *.png                   # Generated plots

├── src/

│   ├── analysis.py             # Compute de alignment score and generate the plots

│   ├── data_loader.py          # Load the data from the API

│   └── embedding.py            # Compute the embedding of the Aim&Scope and the abstracts

├── .env

└── README.md

## How to Run

1. Clone this repository
2. Create a `.env` file with your Semantic Scholar API key: SEMANTIC_SCHOLAR_API_KEY=your_key_here
3. Install dependencies:
```bash
   pip install -r requirements.txt
```
4. Open and run `notebooks/analysis.ipynb`

## Key Results

- **Corpus size:** 330 abstracts (2019-2025)
- **Mean alignment score:** 0.770 (σ = 0.051)
- **Outliers:** 6 low-scoring, 9 high-scoring papers identified
- **Temporal drift:** Limited, with a minor peak in 2023

See the full analysis in [`notebooks/analysis.ipynb`](notebooks/analysis.ipynb) 
and the complete paper in [`paper/sn-article.pdf`](paper/sn-article.pdf).

## Author

Ludovico Maria Passoni  
University of Milan (Unimi)