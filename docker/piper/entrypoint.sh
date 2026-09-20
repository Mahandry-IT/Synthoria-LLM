#!/bin/sh
# Usage : piper-entrypoint serve            → démarre le serveur HTTP
#         piper-entrypoint download <voix…> → télécharge les voix dans /voices puis s'arrête
set -eu

case "${1:-serve}" in
  serve)
    exec python -m piper.http_server \
      -m "${PIPER_DEFAULT_VOICE:-fr_FR-siwis-medium}" \
      --data-dir "${PIPER_DATA_DIR:-/voices}" \
      --host 0.0.0.0 --port 5000
    ;;
  download)
    shift
    exec python -m piper.download_voices "$@" --data-dir "${PIPER_DATA_DIR:-/voices}"
    ;;
  *)
    exec "$@"
    ;;
esac
