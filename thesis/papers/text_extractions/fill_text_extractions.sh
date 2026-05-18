#!/bin/bash -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PAPERS_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

if ! command -v pdftotext >/dev/null 2>&1; then
    echo "pdftotext not found. Install poppler (e.g. 'brew install poppler')." >&2
    exit 1
fi

shopt -s nullglob
rm -f "${SCRIPT_DIR}"/*.txt

for pdf in "${PAPERS_DIR}"/*.pdf; do
    base="$(basename "${pdf}" .pdf)"
    out="${SCRIPT_DIR}/${base}.txt"
    echo "Extracting ${base}.pdf -> ${base}.txt"
    pdftotext -layout "${pdf}" "${out}"
done
