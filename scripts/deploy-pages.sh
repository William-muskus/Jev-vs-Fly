#!/usr/bin/env bash
# Publish web/ (including the exported model in web/model/) to the gh-pages branch.
#
# Usage:  scripts/deploy-pages.sh [remote] [branch]        (defaults: origin gh-pages)
#         DRY_RUN=1 scripts/deploy-pages.sh                (build the commit, do not push)
#
# Only runs when invoked explicitly. Uses a temporary git worktree so the working tree is untouched.
# Commits as Carlo Esposito <96500694+cesp99@users.noreply.github.com>, co-authored by Spettro.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REMOTE="${1:-origin}"
BRANCH="${2:-gh-pages}"
WEB="$ROOT/web"
AUTHOR_NAME="Carlo Esposito"
AUTHOR_EMAIL="96500694+cesp99@users.noreply.github.com"
MAX_FILE_BYTES=$((100 * 1024 * 1024))   # GitHub hard limit per file

cd "$ROOT"

# ---- preconditions -------------------------------------------------------------------------
[[ -f "$WEB/index.html" ]] || { echo "error: $WEB/index.html missing" >&2; exit 1; }
if [[ ! -f "$WEB/model/brain.json" || ! -f "$WEB/model/brain.flyb" ]]; then
  echo "error: no exported model in web/model/ — run 'fly export-web --run <name>' first" >&2
  exit 1
fi
while IFS= read -r -d '' f; do
  size=$(stat -c %s "$f")
  if (( size > MAX_FILE_BYTES )); then
    echo "error: $f is $((size / 1048576)) MB (> 100 MB GitHub limit). Use --quant i8 or drop it." >&2
    exit 1
  fi
done < <(find "$WEB" -type f -print0)
git remote get-url "$REMOTE" >/dev/null 2>&1 || { echo "error: remote '$REMOTE' not configured" >&2; exit 1; }

SRC_COMMIT="$(git rev-parse --short HEAD)"
RUN_NAME="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("run_name","?"))' "$WEB/model/brain.json" 2>/dev/null || echo '?')"

# ---- temporary worktree ---------------------------------------------------------------------
TMP="$(mktemp -d "${TMPDIR:-/tmp}/flychess-pages.XXXXXX")"
cleanup() { git worktree remove --force "$TMP" >/dev/null 2>&1 || rm -rf "$TMP"; git worktree prune >/dev/null 2>&1 || true; }
trap cleanup EXIT

git fetch "$REMOTE" "$BRANCH" >/dev/null 2>&1 || true
if git show-ref --verify --quiet "refs/remotes/$REMOTE/$BRANCH"; then
  git worktree add "$TMP" "$REMOTE/$BRANCH" >/dev/null
  git -C "$TMP" checkout -B "$BRANCH" >/dev/null 2>&1
elif git show-ref --verify --quiet "refs/heads/$BRANCH"; then
  git worktree add "$TMP" "$BRANCH" >/dev/null
else
  git worktree add --detach "$TMP" >/dev/null
  git -C "$TMP" checkout --orphan "$BRANCH" >/dev/null 2>&1
  git -C "$TMP" rm -rfq . >/dev/null 2>&1 || true
fi

# ---- copy site --------------------------------------------------------------------------------
find "$TMP" -mindepth 1 -maxdepth 1 ! -name .git -exec rm -rf {} +
# everything in web/ except node_modules and tests; the model is included on purpose
tar -C "$WEB" --exclude=node_modules --exclude=test --exclude=.DS_Store -cf - . | tar -C "$TMP" -xf -
touch "$TMP/.nojekyll"          # serve files/dirs starting with _ and keep .gz as-is
echo "flychess $SRC_COMMIT model=$RUN_NAME $(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$TMP/BUILD.txt"

# ---- commit & push --------------------------------------------------------------------------
cd "$TMP"
git add -A
if git diff --cached --quiet; then
  echo "nothing to deploy: gh-pages already matches web/"
  exit 0
fi
MSG="Deploy site from $SRC_COMMIT (model: $RUN_NAME)

Co-Authored-By: Spettro <spettro@eyed.to>"
git -c user.name="$AUTHOR_NAME" -c user.email="$AUTHOR_EMAIL" commit -q --author="$AUTHOR_NAME <$AUTHOR_EMAIL>" -m "$MSG"
git log --show-signature -1 --format='%H %G? %an <%ae>' 2>/dev/null | head -1 || true

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "DRY_RUN=1: built commit on '$BRANCH' in $TMP, not pushing."
  trap - EXIT
  echo "(worktree left in place; remove with: git worktree remove --force $TMP)"
  exit 0
fi
git push "$REMOTE" "$BRANCH:$BRANCH"
echo "pushed $BRANCH to $REMOTE — enable GitHub Pages (branch: $BRANCH, folder: /) if not already."
