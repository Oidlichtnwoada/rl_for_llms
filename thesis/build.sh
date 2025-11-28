#!/usr/bin/env bash

set -e

# Optional --clean flag to force full rebuild
FORCE_CLEAN=false
if [[ "$1" == "--clean" ]]; then
  FORCE_CLEAN=true
  echo "⚠️  Clean rebuild requested (-gg enabled)"
fi

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

    LATEXMK_OPTS="-${ENG} -bibtex"

    # Add -gg only on clean builds
    if $FORCE_CLEAN; then
      LATEXMK_OPTS="$LATEXMK_OPTS -gg"
    fi

    docker run --rm \
      -v "$(pwd):/work" \
      -w /work \
      texlive/texlive:latest \
        bash -c "latexmk ${LATEXMK_OPTS} -jobname=%A${SUFFIX} ${SRC}.tex"

    mv "${SRC}${SUFFIX}.pdf" "${OUTPUT_DIR}/"
    if [[ -f "${SRC}${SUFFIX}.log" ]]; then
      mv "${SRC}${SUFFIX}.log" "${OUTPUT_DIR}/"
    fi

    echo "✓ Built: ${OUTPUT_DIR}/${OUTPUT}"
  done
done

echo ""
echo "✅ All builds completed. Output is in the '${OUTPUT_DIR}/' directory."
