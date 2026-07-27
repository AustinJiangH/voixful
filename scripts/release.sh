#!/usr/bin/env bash
#
# Interactive release cutter for the Voixful Swift package.
#
#   make release      # or: bash scripts/release.sh
#
# Walks you through a semantic-version bump (major / minor / patch), gates on a
# clean build + test run, propagates the new version everywhere it is recorded,
# commits + annotated-tags it, and (optionally) pushes and cuts the GitHub
# release. The git tag is the source of truth SwiftPM resolves against; the
# VERSION file and Sources/VoixfulCLI/Version.swift are kept in lockstep so
# `voixful --version` reports the build's version.
#
# macOS / BSD sed assumed (matches the package's macOS-only platform).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

VERSION_FILE="VERSION"
VERSION_SWIFT="Sources/VoixfulCLI/Version.swift"
README="README.md"
REPO_SLUG="AustinJiangH/voixful"

info() { printf '\033[1;36m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m !!\033[0m %s\n' "$*"; }
die()  { printf '\033[1;31mERROR:\033[0m %s\n' "$*" >&2; exit 1; }
ask()  { local p="$1" a; printf '%s' "$p" >&2; read -r a; printf '%s' "$a"; }

# ---------------------------------------------------------------- preconditions
command -v git   >/dev/null || die "git not found"
command -v swift >/dev/null || die "swift toolchain not found"
[ -z "$(git status --porcelain)" ] || die "Working tree is not clean — commit or stash first."

BRANCH="$(git rev-parse --abbrev-ref HEAD)"

# ------------------------------------------------------------- current version
# Prefer the latest tag; fall back to the VERSION file; else 0.0.0.
CUR="$(git describe --tags --abbrev=0 2>/dev/null | sed 's/^v//' || true)"
[ -n "$CUR" ] || CUR="$(cat "$VERSION_FILE" 2>/dev/null || echo '0.0.0')"
echo "$CUR" | grep -Eq '^[0-9]+\.[0-9]+\.[0-9]+$' || die "Can't parse current version '$CUR'"
IFS=. read -r MA MI PA <<<"$CUR"

# ------------------------------------------------------------------ bump choice
info "Current version: v$CUR   (branch: $BRANCH)"
echo "Select release type:"
echo "  1) patch  ->  $MA.$MI.$((PA + 1))    (bug fixes, no API change)"
echo "  2) minor  ->  $MA.$((MI + 1)).0    (new, backward-compatible features)"
echo "  3) major  ->  $((MA + 1)).0.0    (breaking changes)"
CHOICE="$(ask 'Choice [1-3] (default 1): ')"
case "${CHOICE:-1}" in
  1) NEW="$MA.$MI.$((PA + 1))" ;;
  2) NEW="$MA.$((MI + 1)).0" ;;
  3) NEW="$((MA + 1)).0.0" ;;
  *) die "Invalid choice: $CHOICE" ;;
esac
TAG="v$NEW"

git rev-parse -q --verify "refs/tags/$TAG" >/dev/null && die "Tag $TAG already exists."

CONFIRM="$(ask "Cut release $TAG from $BRANCH? [y/N]: ")"
[ "$CONFIRM" = y ] || [ "$CONFIRM" = Y ] || die "Aborted."

# --------------------------------------------------------------- build + tests
info "Building (release)…"
swift build -c release
info "Running tests…"
swift test

# ------------------------------------------------------- propagate the version
info "Updating $VERSION_FILE"
echo "$NEW" > "$VERSION_FILE"

info "Updating $VERSION_SWIFT"
cat > "$VERSION_SWIFT" <<EOF
// Generated / bumped by scripts/release.sh (via \`make release\`) — do not edit by hand.
// The single source of truth for the released version is the git tag; the VERSION
// file and this constant are kept in lockstep with it so \`voixful --version\`
// reports the build's version. "0.0.0" means an unreleased working tree.
let voixfulVersion = "$NEW"
EOF

info "Pinning the README dependency snippet to $NEW"
if grep -q 'voixful\.git", branch: "main"' "$README"; then
  sed -i '' 's#\(voixful\.git", \)branch: "main"#\1from: "'"$NEW"'"#' "$README"
elif grep -Eq 'voixful\.git", from: "[0-9]+\.[0-9]+\.[0-9]+"' "$README"; then
  sed -i '' 's#\(voixful\.git", from: "\)[0-9.]*"#\1'"$NEW"'"#' "$README"
else
  warn "Couldn't find the dependency line to pin in $README — update it by hand."
fi

# ------------------------------------------------------------- commit + tag
info "Committing and tagging $TAG"
git add "$VERSION_FILE" "$VERSION_SWIFT" "$README"
git commit -m "Release $TAG" -m "Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
git tag -a "$TAG" -m "Voixful $TAG"

# ------------------------------------------------------------------- publish?
PUSH="$(ask "Push $BRANCH and $TAG to origin? [y/N]: ")"
if [ "$PUSH" = y ] || [ "$PUSH" = Y ]; then
  git push origin "$BRANCH"
  git push origin "$TAG"
  if command -v gh >/dev/null; then
    GHR="$(ask "Create the GitHub release for $TAG (auto-generated notes)? [y/N]: ")"
    if [ "$GHR" = y ] || [ "$GHR" = Y ]; then
      gh release create "$TAG" --repo "$REPO_SLUG" --title "$TAG" --generate-notes
    fi
  else
    warn "gh not installed — create the GitHub release manually if you want one."
  fi
else
  warn "Not pushed. Publish later with:"
  warn "    git push origin $BRANCH && git push origin $TAG"
fi

info "Done — $TAG is tagged locally."
