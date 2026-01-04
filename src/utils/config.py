# src/utils/config.py

import os

# --------------------------
# Configuration Binance API
# --------------------------
BINANCE_API_URL = "https://api.binance.com/api/v3"

# Comme nous utilisons seulement les endpoints publics, pas besoin de clé API
BINANCE_API_KEY = None
BINANCE_SECRET_KEY = None

# --------------------------
# Chemins de fichiers / dossiers
# --------------------------
# Répertoire racine pour les données brutes
RAW_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "..", "data", "raw")

# Répertoire pour les données transformées
PROCESSED_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "..", "data", "processed")

# Répertoire pour les exemples de données
EXAMPLES_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "..", "data", "examples")

# Répertoire pour les logs
LOGS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "..", "logs")

# --------------------------
# Paramètres généraux
# --------------------------
# Timeout des requêtes HTTP (en secondes)
REQUEST_TIMEOUT = 10

# Nombre maximum d'éléments à récupérer par requête pour certains endpoints
MAX_LIMIT = 1000

# Intervalle par défaut pour les Klines si non spécifié
DEFAULT_KLINE_INTERVAL = "1h"

# Active ou non le mode debug (affichage des logs sur console)
DEBUG = True
