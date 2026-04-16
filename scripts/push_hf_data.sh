#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HF_REPO_DIR="${HF_REPO_DIR:-$ROOT_DIR/hf-data}"
HF_SUBDIR="${HF_SUBDIR:-artifacts}"
COMMIT_MSG="${COMMIT_MSG:-sync data artifacts from main repo}"

if [[ ! -d "$HF_REPO_DIR/.git" ]]; then
  echo "HF repo dir not found: $HF_REPO_DIR" >&2
  exit 1
fi

if [[ $# -eq 0 ]]; then
  echo "Usage: $0 <path> [path ...]" >&2
  echo "Example: $0 data/nld_large_run/eval_tail_1024.jsonl output/qwen14b_nld_long_32k_run" >&2
  exit 1
fi

cd "$ROOT_DIR"

mkdir -p "$HF_REPO_DIR/$HF_SUBDIR"

git -C "$HF_REPO_DIR" lfs track "*.jsonl" "*.json" "*.safetensors" "*.bin" "*.pt" "*.pth" "*.db" >/dev/null

if [[ ! -f "$HF_REPO_DIR/README.md" ]]; then
  cat > "$HF_REPO_DIR/README.md" <<'EOF'
# rl-nethack-data

Large data and artifact mirror for the `rl-nethack` project.

This repo is intended for Hugging Face Hub storage via Git LFS.
EOF
fi

for src in "$@"; do
  if [[ ! -e "$src" ]]; then
    echo "Missing source path: $src" >&2
    exit 1
  fi
  dest="$HF_REPO_DIR/$HF_SUBDIR/$src"
  mkdir -p "$(dirname "$dest")"
  if [[ -d "$src" ]]; then
    rsync -a --delete "$src"/ "$dest"/
  else
    rsync -a "$src" "$dest"
  fi
done

git -C "$HF_REPO_DIR" add .gitattributes README.md "$HF_SUBDIR"

if git -C "$HF_REPO_DIR" diff --cached --quiet; then
  echo "No changes to push."
  exit 0
fi

git -C "$HF_REPO_DIR" commit -m "$COMMIT_MSG"
git -C "$HF_REPO_DIR" push origin HEAD
