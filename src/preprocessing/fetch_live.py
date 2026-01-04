# ----------------------------
# Import des bibliothèques
# ----------------------------

import requests                            # Pour effectuer des requêtes HTTP vers l’API Binance
import json                                # Pour sauvegarder les données au format JSON
from datetime import datetime, timezone    # Pour horodater la récupération des données
from pathlib import Path                   # Pour gérer les chemins de fichiers proprement

# ----------------------------
# Constantes globales
# ----------------------------

# URL de l’API Binance permettant de récupérer les statistiques 24h d’un symbole
BASE_URL = "https://api.binance.com/api/v3/ticker/24hr"

# Dossier où seront stockées les données brutes (raw) en temps réel (live)
RAW_LIVE_PATH = Path("data/raw/live")

# Création du dossier s’il n’existe pas déjà
# parents=True : crée les dossiers parents si nécessaire
# exist_ok=True : évite une erreur si le dossier existe déjà
RAW_LIVE_PATH.mkdir(parents=True, exist_ok=True)

# ----------------------------
# Fonctions
# ----------------------------

def fetch_live_symbol(symbol: str) -> dict:
    """
    Récupère les données live d'un symbole Binance à l'instant T.

    Args:
        symbol (str): Symbole Binance (ex: "BTCUSDT")

    Returns:
        dict: Données du symbole enrichies avec l'heure de récupération
    """

    # Envoi d'une requête GET à l'API Binance avec le symbole en paramètre
    response = requests.get(BASE_URL, params={"symbol": symbol})

    # Lève une exception si la requête HTTP a échoué (code != 200)
    response.raise_for_status()

    # Conversion de la réponse JSON en dictionnaire Python
    data = response.json()

    # Ajout de l'heure exacte de récupération (UTC) pour traçabilité
    data["retrieval_time"] = datetime.now(timezone.utc).isoformat()


    return data


def save_live_data(symbol: str, data: dict):
    """
    Sauvegarde les données live d'un symbole dans un fichier JSON.

    Args:
        symbol (str): Symbole Binance
        data (dict): Données à sauvegarder
    """

    # Nom du fichier : ex btcusdt_live.json
    filename = RAW_LIVE_PATH / f"{symbol.lower()}_live_[{data['retrieval_time']}].json"

    # Écriture des données au format JSON, avec indentation pour lisibilité
    with open(filename, "w") as f:
        json.dump(data, f, indent=4)

    # Message de confirmation dans la console
    print(f"[LIVE] Données sauvegardées : {filename}")

# ----------------------------
# Point d’entrée du script
# ----------------------------

if __name__ == "__main__":
    """
    Ce bloc s'exécute uniquement si le fichier est lancé directement
    (et non importé comme module).
    """

    # Liste des symboles à surveiller
    symbols = ["BTCUSDT", "ETHUSDT"]

    # Boucle sur chaque symbole
    for symbol in symbols:
        # Récupération des données live
        live_data = fetch_live_symbol(symbol)

        # Sauvegarde des données sur disque
        save_live_data(symbol, live_data)
