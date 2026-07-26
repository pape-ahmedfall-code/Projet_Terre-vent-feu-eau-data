# Image de base Python 3.11 légère
FROM python:3.11-slim

# Empêcher Python d'écrire des fichiers .pyc et forcer l'affichage immédiat des logs
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Définir le répertoire de travail dans le conteneur
WORKDIR /app

# Installer les dépendances système nécessaires pour PyTorch / OpenCV / SciPy si besoin
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /lib/apt/lists/*

# Copier le fichier des dépendances
COPY requirements.txt .

# Installer les packages Python
RUN pip install --no-cache-dir -r requirements.txt

# Copier l'intégralité du code du projet dans le conteneur
COPY . .

# Exposer le port par défaut de Streamlit
EXPOSE 8501

# Commande de lancement de l'application Streamlit
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]