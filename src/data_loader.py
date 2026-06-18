import requests
import pandas as pd
import time
from dotenv import load_dotenv
import os
import json

load_dotenv()

API_KEY = os.getenv("API_KEY")

BASE_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
PARAMS = {
    "query": "EMNLP empirical methods natural language processing",
    "fields": "title,abstract,year,venue",
    "limit": 100
}

HEADERS = {"x-api-key": API_KEY}


articoli_validi = []
offset = 0

while len(articoli_validi) < 1000:
    PARAMS["offset"] = offset
    
    response = requests.get(BASE_URL, headers=HEADERS, params=PARAMS)   #Chiamata all'API
    data = response.json()                                              #Trasforma quello che riceve in un file json
    if response.status_code == 429:
        print("Troppe richieste, aspetto 10 secondi...")
        time.sleep(10)
        continue
    
    if "data" not in data or len(data["data"]) == 0:                    #La risposta non contiene il campo API o pagina vuota
        break
    
    for articolo in data["data"]:
        venue = articolo.get("venue", "")                               #prendo da che rivista deriva l'articolo, se non c'è restituisco una stringa vuota
        abstract = articolo.get("abstract", None)                       #prendo il contenuto dell'articolo, se non c'è restituisco None
        year = articolo.get("year", None)                               #prendo l'anno di pubblicazione, se non c'è restituisco None

        if venue == "Conference on Empirical Methods in Natural Language Processing" and abstract is not None and year is not None and year >= 2019:                   #controllo che la rivista sia quella giusta e che l'articolo abbia un contenuto
            articoli_validi.append({                                    #aggiugo l'articolo alla lista di articoli validi
                "title": articolo.get("title", ""),
                "abstract": abstract,
                "year": articolo.get("year", ""),
                "venue": venue
            })
    
    offset += 100                                                          #aggiorno l'ofsett per prendere i nuovi articoli
    time.sleep(3)                                                          #aspetto un secondo per evitare di sovraccaricare l'API
    print(f"Offset: {offset} | Articoli validi finora: {len(articoli_validi)}")

print(f"Articoli raccolti: {len(articoli_validi)}")                        #numero di articoli ottenuti

with open("data/abstracts.json", "w") as f:
    json.dump(articoli_validi, f, indent=2)                                #salvo i dati nel file json in modo leggibile

print("Salvato in data/abstracts.json")