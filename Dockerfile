# 1. Usa un'immagine ufficiale di Python leggera (slim)
FROM python:3.12-slim

# 2. Imposta la cartella di lavoro all'interno del container
WORKDIR /app

# --- AGGIUNTA: Installa le librerie di sistema necessarie per LightGBM ---
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*
# ------------------------------------------------------------------------

# 3. Copia il file dei requisiti e installa le dipendenze
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copia il codice dell'API e i file .pkl del modello
COPY app2.py .
COPY venom_triage_pipeline.pkl .
COPY label_encoder_target.pkl .

# 5. Esponi la porta 8000
EXPOSE 8000

# 6. Comando per avviare Uvicorn
CMD ["uvicorn", "app2:app", "--host", "0.0.0.0", "--port", "8000"]