# Serveur de synthèse vocale Piper (OHF-Voice/piper1-gpl, licence GPL-3.0).
# Isolé dans son propre conteneur : l'API et le worker l'appellent en HTTP uniquement.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIPER_DATA_DIR=/voices

RUN pip install --no-cache-dir "piper-tts[http]>=1.3,<2"

RUN useradd --create-home --uid 1000 piper \
    && mkdir -p /voices \
    && chown -R piper:piper /voices
COPY docker/piper/entrypoint.sh /usr/local/bin/piper-entrypoint
RUN chmod +x /usr/local/bin/piper-entrypoint

USER piper
EXPOSE 5000
ENTRYPOINT ["piper-entrypoint"]
