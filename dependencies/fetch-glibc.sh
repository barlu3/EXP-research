#!/usr/bin/env bash
# Fetches glibc-2.43 source tarball + signature into this directory.
# Extract with: tar xf glibc-2.43.tar.xz
set -euo pipefail

GLIBC_VERSION="2.43"
MIRROR="https://ftp.gnu.org/gnu/glibc"
TARBALL="glibc-${GLIBC_VERSION}.tar.xz"
SIGFILE="${TARBALL}.sig"

cd "$(dirname "$(readlink -f "$0")")"

fetch() {
    local url="$1" out="$2"
    if [ -f "$out" ]; then
        echo "have ${out}, skipping"
        return
    fi
    echo "fetching ${url}"
    if command -v curl >/dev/null 2>&1; then
        curl -fL --retry 3 -o "${out}.part" "$url"
    elif command -v wget >/dev/null 2>&1; then
        wget -O "${out}.part" "$url"
    else
        echo "error: need curl or wget" >&2
        exit 1
    fi
    mv "${out}.part" "$out"
}

fetch "${MIRROR}/${TARBALL}" "$TARBALL"
fetch "${MIRROR}/${SIGFILE}" "$SIGFILE"

# NOTE: GNU release keys are not in a stock keyring, so signature check is
# best-effort and non-fatal. Import the key with:
#   gpg --keyserver keyserver.ubuntu.com --recv-keys <KEYID from gpg --verify output>
if command -v gpg >/dev/null 2>&1; then
    if gpg --verify "$SIGFILE" "$TARBALL" 2>/dev/null; then
        echo "signature OK"
    else
        echo "warning: signature not verified (missing GNU release key?)" >&2
    fi
else
    echo "warning: gpg not found, skipping signature check" >&2
fi

echo "done: ${TARBALL}"
