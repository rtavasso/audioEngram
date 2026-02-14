#!/bin/bash
# Download LibriSpeech train-clean-100 subset
# Source: https://www.openslr.org/12/

set -e

DATA_DIR="${1:-./data}"
SUBSET="train-clean-100"
URL="https://www.openslr.org/resources/12/${SUBSET}.tar.gz"

echo "Downloading LibriSpeech ${SUBSET}..."
echo "Data directory: ${DATA_DIR}"

# Create data directory
mkdir -p "${DATA_DIR}"
cd "${DATA_DIR}"

# Download if not exists
if [ ! -f "${SUBSET}.tar.gz" ]; then
    echo "Downloading from ${URL}..."
    wget -c "${URL}"
else
    echo "Archive already exists, skipping download."
fi

# Extract if not already extracted
if [ ! -d "LibriSpeech/${SUBSET}" ]; then
    echo "Extracting ${SUBSET}.tar.gz..."
    tar -xzf "${SUBSET}.tar.gz"
else
    echo "Already extracted, skipping."
fi

# Verify
N_SPEAKERS=$(ls -d LibriSpeech/${SUBSET}/*/ 2>/dev/null | wc -l)
echo ""
echo "Download complete!"
echo "Location: ${DATA_DIR}/LibriSpeech/${SUBSET}"
echo "Number of speakers: ${N_SPEAKERS}"

# Count total audio files
N_FILES=$(find "LibriSpeech/${SUBSET}" -name "*.flac" | wc -l)
echo "Number of audio files: ${N_FILES}"
