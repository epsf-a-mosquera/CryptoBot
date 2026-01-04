# ============================================================
# src/data_collection/fetch_historical.py
# Objectif : Récupérer les données historiques des crypto-monnaies
#            depuis Binance et les sauvegarder au format JSON.
# ============================================================

"""
Ce script permet de télécharger l'historique complet des bougies
(Klines) pour une liste de symboles depuis Binance. 

Les fichiers JSON sont stockés dans le dossier : data/raw/historical/
Chaque fichier contient toutes les bougies disponibles depuis la date
de début définie.
"""

# ----------------------------
# Import des bibliothèques
# ----------------------------

import os          # Gestion des dossiers et chemins
import json        # Pour sauvegarder les données au format JSON
import time        # Pour faire des pauses entre les requêtes
import requests    # Pour interroger l'API Binance
from datetime import datetime  # Pour gérer les dates

# ----------------------------
# Configuration
# ----------------------------

# Liste des symboles à récupérer (ex : BTC/USDT, ETH/USDT)
SYMBOLS = ["BTCUSDT", "ETHUSDT"]

# Intervalle des bougies (klines) : '1m', '5m', '1h', '1d', etc.
INTERVAL = "1h"

# Date de départ pour récupérer l'historique
START_DATE = "2022-01-01"

# Dossier pour sauvegarder les fichiers JSON
OUTPUT_DIR = "data/raw/historical/"

# Nombre maximum de bougies retournées par requête (limite Binance = 1000)
LIMIT = 1000

# URL de base de l'API Binance pour récupérer les klines
BASE_URL = "https://api.binance.com/api/v3/klines"

# ----------------------------
# Fonctions utilitaires
# ----------------------------

def date_to_milliseconds(date_str):
    """
    Convertit une date (format 'YYYY-MM-DD') en timestamp en millisecondes.
    Binance API utilise les timestamps en ms.
    
    Args:
        date_str (str) : Date sous forme 'YYYY-MM-DD'
    
    Returns:
        int : Timestamp en millisecondes
    """
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    return int(dt.timestamp() * 1000)


def fetch_klines(symbol, interval, start_time, limit=1000):
    """
    Récupère les bougies (klines) depuis Binance pour un symbole donné.
    Cette fonction gère la pagination automatiquement pour récupérer
    tout l'historique disponible depuis la date de départ.
    
    Args:
        symbol (str) : Symbole Binance (ex: 'BTCUSDT')
        interval (str) : Intervalle des bougies ('1m', '5m', '1h', '1d', ...)
        start_time (int) : Timestamp de départ en millisecondes
        limit (int) : Nombre maximum de bougies par requête (max 1000)
    
    Returns:
        list : Liste de toutes les bougies récupérées
    """

    all_klines = []  # Liste pour stocker toutes les bougies
    start_ts = start_time  # Timestamp courant pour la pagination

    while True:
        # Paramètres de la requête
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": start_ts,
            "limit": limit
        }

        # Requête à l'API Binance
        response = requests.get(BASE_URL, params=params)
        data = response.json()

        # Si aucune donnée, fin de la récupération
        if not data:
            break

        # Ajout des bougies récupérées à la liste complète
        all_klines.extend(data)

        # Mise à jour du timestamp pour la prochaine requête
        # +1 ms pour éviter les doublons
        start_ts = data[-1][0] + 1

        # Si le nombre de bougies retournées est inférieur à LIMIT, c'est la fin
        if len(data) < limit:
            break

        # Pause courte pour ne pas saturer l'API et éviter les erreurs de rate limit
        time.sleep(1)

    return all_klines


def save_to_json(symbol, klines):
    """
    Sauvegarde les bougies dans un fichier JSON.
    
    Args:
        symbol (str) : Symbole Binance
        klines (list) : Liste des bougies récupérées
    """
    # Création du dossier si inexistant
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # Chemin du fichier JSON
    file_path = os.path.join(OUTPUT_DIR, f"{symbol}_{INTERVAL}.json")

    # Écriture des données dans le fichier avec indentation pour lisibilité
    with open(file_path, "w") as f:
        json.dump(klines, f, indent=4)

    # Message d'information
    print(f"[INFO] {symbol} : {len(klines)} bougies sauvegardées dans {file_path}")


# ----------------------------
# Script principal
# ----------------------------

if __name__ == "__main__":

    # Conversion de la date de départ en timestamp millisecondes
    start_ms = date_to_milliseconds(START_DATE)

    # Boucle sur chaque symbole de la liste
    for symbol in SYMBOLS:
        print(f"[INFO] Récupération de l'historique pour {symbol} depuis {START_DATE}")

        # Récupération des bougies depuis Binance
        klines = fetch_klines(symbol, INTERVAL, start_ms, LIMIT)

        # Sauvegarde des bougies dans un fichier JSON
        save_to_json(symbol, klines)

    print("[INFO] Récupération terminée !")
