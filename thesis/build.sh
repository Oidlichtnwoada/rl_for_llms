#!/usr/bin/env bash

set -e

# Sources and engines
SOURCES=("main-thesis" "main-report" "main-seminarreport")
ENGINES=("pdf" "pdfxe" "pdflua")

OUTPUT_DIR="build"
mkdir -p "${OUTPUT_DIR}"

echo "=== Local LaTeX Build Script ==="

map_engine_to_command() {
  case "$1" in
    pdf) echo "pdflatex" ;;
    pdfxe) echo "xelatex" ;;
    pdflua) echo "lualatex" ;;
  esac
}

map_engine_to_suffix() {
  case "$1" in
    pdflatex) echo "-pdflatex" ;;
    xelatex) echo "-xelatex" ;;
    lualatex) echo "-lualatex" ;;
  esac
}

for SRC in "${SOURCES[@]}"; do
  for ENG in "${ENGINES[@]}"; do

    ENGINE_CMD=$(map_engine_to_command "$ENG")
    SUFFIX=$(map_engine_to_suffix "$ENGINE_CMD")

    OUTPUT="${SRC}${SUFFIX}.pdf"
    LOG="${SRC}${SUFFIX}.log"

    echo ""
    echo ">>> Building $SRC.tex using $ENGINE_CMD …"

    docker run --rm -v "$(pwd):/work" -w /work texlive/texlive:latest bash -c "latexmk -${ENG} -bibtex -gg -jobname=%A${SUFFIX} ${SRC}.tex"

    mv "${SRC}${SUFFIX}.pdf" "${OUTPUT_DIR}/"
    mv "${SRC}${SUFFIX}.log" "${OUTPUT_DIR}/"

    echo "✓ Built: ${OUTPUT_DIR}/${OUTPUT}"
  done
done

echo ""
echo "✅ All builds completed. Output is in the '${OUTPUT_DIR}/' directory."
