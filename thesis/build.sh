#!/usr/bin/env bash

set -e

# Start the timer
START_TIME=$SECONDS

# --- Configuration ---
IMAGE="texlive/texlive:latest"
OUTPUT_DIR="build"
DEFAULT_SOURCE="main-thesis"
ENGINES_TO_RUN=("pdf") # Default engine
SOURCES=()
FORCE_CLEAN=false
SKIP_IMAGE_CHECK=false

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
    echo "                    Available: pdf (default), xe"
    echo "                    Example: --engine=pdf,xe"
    echo "  --skip-image-check Do not verify that $IMAGE is the newest"
    echo "                    published revision (useful when offline)"
    echo "  --help            Show this help message"
    echo ""
    echo "If no source files are provided, defaults to: $DEFAULT_SOURCE"
    echo ""
    echo "Sources are LaTeX root files without the .tex suffix, e.g.:"
    echo "  $0 main-thesis          Build the thesis (default)"
    echo "  $0 main-poster          Build the A0 landscape poster"
    echo "  $0 main-presentation    Build the 16:9 defense presentation"
    echo "  $0 main-thesis main-poster   Build both"
}

# Resolve the digest that the registry currently serves for $IMAGE. Prints the
# digest on success and nothing on failure (offline, rate-limited, ...), so the
# caller decides how to treat an unknown remote state.
remote_image_digest() {
    local image="$1"

    # Preferred path: buildx ships with every modern Docker and prints the
    # manifest digest directly, without pulling any layer.
    if docker buildx version >/dev/null 2>&1; then
        docker buildx imagetools inspect "$image" \
            --format '{{.Manifest.Digest}}' 2>/dev/null && return 0
    fi

    # Fallback for installations without buildx: the verbose manifest carries
    # the same digest under .Descriptor.digest.
    docker manifest inspect --verbose "$image" 2>/dev/null \
        | sed -n 's/.*"digest"[[:space:]]*:[[:space:]]*"\(sha256:[0-9a-f]*\)".*/\1/p' \
        | head -n 1
}

# Pull $1, and if that fails try to make room and pull once more. A multi-GB
# TeX Live image regularly fails on "no space left on device", and the space is
# almost always sitting in Docker's regenerable caches, so reclaim those rather
# than making the user diagnose it. Only caches are touched: named volumes and
# tagged images belonging to other projects are never removed.
pull_with_remediation() {
    local image="$1"

    if docker pull "$image"; then
        return 0
    fi

    echo -e "${YELLOW}⚠ Pull failed. Reclaiming Docker cache space and retrying...${NC}"

    # Dangling build cache first: it is pure scratch space and usually the
    # single largest consumer.
    docker builder prune --force >/dev/null 2>&1 \
        && echo -e "${YELLOW}  Reclaimed the build cache.${NC}"
    # Then untagged (dangling) image layers, e.g. the superseded revisions of
    # this very tag. Tagged images are left alone.
    docker image prune --force >/dev/null 2>&1 \
        && echo -e "${YELLOW}  Reclaimed dangling image layers.${NC}"

    docker pull "$image"
}

# Make sure the locally cached $IMAGE really is the revision the registry
# publishes right now. A "latest" tag pinned weeks ago silently keeps building
# against an outdated TeX Live, so compare digests and re-pull on a mismatch.
ensure_latest_image() {
    local image="$1"

    echo -e "${BLUE}=== Checking ${image} is up to date ===${NC}"

    if ! command -v docker >/dev/null 2>&1; then
        echo -e "${RED}Error: docker is not installed or not on PATH.${NC}"
        exit 1
    fi

    if ! docker image inspect "$image" >/dev/null 2>&1; then
        echo -e "${YELLOW}Image not present locally. Pulling...${NC}"
        pull_with_remediation "$image" \
            || { echo -e "${RED}✗ Cannot obtain ${image}. Aborting.${NC}"; exit 1; }
        return
    fi

    local remote_digest
    remote_digest="$(remote_image_digest "$image")"

    if [[ -z "$remote_digest" ]]; then
        echo -e "${YELLOW}⚠ Could not reach the registry. Building against the cached image.${NC}"
        return
    fi

    # A single local tag can carry several RepoDigests (one per registry it was
    # pushed to/pulled from), so test for membership rather than equality.
    local local_digests
    local_digests="$(docker image inspect \
        --format '{{range .RepoDigests}}{{println .}}{{end}}' "$image" 2>/dev/null)"

    if grep -qF "$remote_digest" <<< "$local_digests"; then
        echo -e "${GREEN}✓ Up to date (${remote_digest}).${NC}"
        return
    fi

    local local_created
    local_created="$(docker image inspect --format '{{.Created}}' "$image" 2>/dev/null)"
    echo -e "${YELLOW}⚠ Local image is outdated (built ${local_created%%T*}).${NC}"
    echo -e "${YELLOW}  Registry now serves ${remote_digest}. Pulling the new revision...${NC}"

    if pull_with_remediation "$image"; then
        echo -e "${GREEN}✓ Replaced the local image with the current ${image}.${NC}"
    else
        # The cached image still builds every document, so a failed refresh must
        # not block the build: warn clearly and carry on with what we have.
        echo -e "${YELLOW}⚠ Could not refresh ${image}; building against the outdated${NC}"
        echo -e "${YELLOW}  local copy from ${local_created%%T*}. Free some disk space and${NC}"
        echo -e "${YELLOW}  re-run to pick up the new TeX Live.${NC}"
    fi
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
        --skip-image-check)
            SKIP_IMAGE_CHECK=true
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

# --- Step 0: Toolchain image freshness ---

if $SKIP_IMAGE_CHECK; then
    echo -e "${YELLOW}Skipping the ${IMAGE} freshness check (--skip-image-check).${NC}"
else
    ensure_latest_image "$IMAGE"
fi

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
            *)
                echo -e "${RED}Error: Unknown or unsupported engine '$ENG'. Skipping.${NC}"
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
          -v "$(pwd)/..:/work" \
          -w /work/thesis \
          -u "$(id -u):$(id -g)" \
          "$IMAGE" \
            bash -c "latexmk ${CMD_OPTS} \"${SRC_FILE}\""

        EXIT_CODE=$?

        # Check if build succeeded
        if [[ $EXIT_CODE -eq 0 && -f "$FINAL_PDF" ]]; then
            echo -e "${GREEN}✓ Built successfully: ${FINAL_PDF}${NC}"
            # Clean up artifacts that latexmk might have generated
            rm -f "${JOBNAME}.pdf" \
                  "${JOBNAME}.aux" "${JOBNAME}.log" "${JOBNAME}.out" "${JOBNAME}.toc" \
                  "${JOBNAME}.fls" "${JOBNAME}.fdb_latexmk" "${JOBNAME}.blg" "${JOBNAME}.bbl" \
                  "${JOBNAME}.synctex.gz" "${JOBNAME}.xdv"
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