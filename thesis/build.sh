#!/usr/bin/env bash

set -e

# Start the timer
START_TIME=$SECONDS

# --- Configuration ---
IMAGE="texlive/texlive:latest"
OUTPUT_DIR="build"
DEFAULT_SOURCE="main-seminarreport"
ENGINES_TO_RUN=("pdf") # Default engine
SOURCES=()
FORCE_CLEAN=false

# ANSI Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# --- Helper Functions ---

print_usage() {
    echo "Usage: $0 [options] [source_files...]"
    echo ""
    echo "Options:"
    echo "  --clean           Force a full rebuild (latexmk -gg)"
    echo "  --engine=TYPE     Select engine(s) to run (comma separated)."
    echo "                    Available: pdf (default), xe, lua"
    echo "                    Example: --engine=pdf,xe"
    echo "  --help            Show this help message"
    echo ""
    echo "If no source files are provided, defaults to: $DEFAULT_SOURCE"
}

# --- Argument Parsing ---

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --clean)
            FORCE_CLEAN=true
            shift
            ;;
        --engine=*)
            IFS=',' read -ra INPUT_ENGINES <<< "${1#*=}"
            ENGINES_TO_RUN=() # Clear default
            for eng in "${INPUT_ENGINES[@]}"; do
                ENGINES_TO_RUN+=("$eng")
            done
            shift
            ;;
        --help)
            print_usage
            exit 0
            ;;
        -*)
            echo -e "${RED}Unknown option: $1${NC}"
            print_usage
            exit 1
            ;;
        *)
            SOURCES+=("$1")
            shift
            ;;
    esac
done

# Set default source if none provided
if [[ ${#SOURCES[@]} -eq 0 ]]; then
    SOURCES=("$DEFAULT_SOURCE")
fi

# Create build directory
mkdir -p "$OUTPUT_DIR"

# --- Step 1: Formatting ---

echo -e "${BLUE}=== Running latexindent.pl ===${NC}"

# Enable nullglob so loop doesn't run on literal "*.sty" if no match exists
shopt -s nullglob

FILES_TO_FORMAT=()
PATTERNS=("*.tex" "*.bbx" "*.bst" "*.cbx" "*.dbx" "*.sty" "*.bib")

for pattern in "${PATTERNS[@]}"; do
    for file in $pattern; do
        [[ -f "$file" ]] && FILES_TO_FORMAT+=("$file")
    done
done

# Disable nullglob to return to normal bash behavior
shopt -u nullglob

if [[ ${#FILES_TO_FORMAT[@]} -gt 0 ]]; then
    echo "Formatting ${#FILES_TO_FORMAT[@]} files..."

    # Run Docker ONCE for all files
    docker run --rm \
      -v "$(pwd):/work" \
      -w /work \
      -u "$(id -u):$(id -g)" \
      "$IMAGE" bash -c "latexindent -w -s ${FILES_TO_FORMAT[*]}"

    # Remove backup and log files created by latexindent
    rm -f *.bak0 indent.log
    echo -e "${GREEN}✓ Formatting complete.${NC}"
else
    echo -e "${YELLOW}No matching files found to format.${NC}"
fi

# --- Step 2: Building ---

echo -e "\n${BLUE}=== Local LaTeX Build Script ===${NC}"

for INPUT_SRC in "${SOURCES[@]}"; do

    # Handle extension: If user typed "main", treat as "main.tex"
    if [[ "$INPUT_SRC" == *.tex ]]; then
        SRC_FILE="$INPUT_SRC"
        SRC_NAME="${INPUT_SRC%.*}"
    else
        SRC_FILE="${INPUT_SRC}.tex"
        SRC_NAME="$INPUT_SRC"
    fi

    # Check existence before Docker
    if [[ ! -f "$SRC_FILE" ]]; then
        echo -e "${RED}Error: Source file '$SRC_FILE' not found. Skipping.${NC}"
        continue
    fi

    for ENG in "${ENGINES_TO_RUN[@]}"; do

        # Map user input to latexmk flags and suffixes
        case "$ENG" in
            pdf)
                LATEXMK_FLAG="-pdf"
                SUFFIX=""
                ;;
            xe|xetex|pdfxe)
                LATEXMK_FLAG="-pdfxe"
                SUFFIX="-xelatex"
                ;;
            lua|lualatex|pdflua)
                LATEXMK_FLAG="-pdflua"
                SUFFIX="-lualatex"
                ;;
            *)
                echo -e "${RED}Error: Unknown engine '$ENG'. Skipping.${NC}"
                continue
                ;;
        esac

        JOBNAME="${SRC_NAME}${SUFFIX}"
        # Because we use -outdir, the file ends up in OUTPUT_DIR automatically
        FINAL_PDF="${OUTPUT_DIR}/${JOBNAME}.pdf"

        echo -e "\n>>> Building ${YELLOW}${SRC_FILE}${NC} using ${YELLOW}${ENG}${NC}..."

        # -outdir keeps root clean; -interaction=nonstopmode prevents hanging on errors
        CMD_OPTS="$LATEXMK_FLAG -bibtex -outdir=$OUTPUT_DIR -jobname=$JOBNAME -interaction=nonstopmode"

        if $FORCE_CLEAN; then
            CMD_OPTS="$CMD_OPTS -gg"
        fi

        # Run compilation
        docker run --rm \
          -v "$(pwd):/work" \
          -w /work \
          -u "$(id -u):$(id -g)" \
          "$IMAGE" \
            bash -c "latexmk ${CMD_OPTS} \"${SRC_FILE}\""

        EXIT_CODE=$?

        # Check if build succeeded
        if [[ $EXIT_CODE -eq 0 && -f "$FINAL_PDF" ]]; then
            echo -e "${GREEN}✓ Built successfully: ${FINAL_PDF}${NC}"
        else
            echo -e "${RED}✗ Build failed for ${SRC_FILE} ($ENG). Check logs in ${OUTPUT_DIR}/${JOBNAME}.log${NC}"
        fi
    done
done

# --- Step 3: Summary ---

DURATION=$(( SECONDS - START_TIME ))
MINUTES=$(( DURATION / 60 ))
SECONDS_REM=$(( DURATION % 60 ))

echo -e "\n${BLUE}=======================================${NC}"
echo -e " ${GREEN}✅ Tasks completed.${NC}"
echo -e " 📂  Output directory: ${YELLOW}${OUTPUT_DIR}/${NC}"
echo -e " ⏱️  Total Duration:   ${YELLOW}${MINUTES}m ${SECONDS_REM}s${NC}"