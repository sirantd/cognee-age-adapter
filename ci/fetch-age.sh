#!/usr/bin/env bash
# Lay out the AGE files of the pinned PGDG package as production mounts them:
#   <dest>/lib/age.so
#   <dest>/share/extension/age.control and age--*.sql
# Usage: ci/fetch-age.sh <dest>
set -euo pipefail

dest=${1:?usage: fetch-age.sh <dest>}
here=$(cd "$(dirname "$0")" && pwd)
# shellcheck source=pins.env
source "$here/pins.env"

work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT

curl -fsSL -o "$work/age.deb" "$AGE_DEB_URL"
echo "$AGE_DEB_SHA256  $work/age.deb" | sha256sum -c -

# ar and tar, not dpkg-deb, so the script also runs on a Mac.
(cd "$work" && ar x age.deb && mkdir root && tar -xJf data.tar.xz -C root)

mkdir -p "$dest/lib" "$dest/share/extension"
cp "$work/root/usr/lib/postgresql/18/lib/age.so" "$dest/lib/"
cp "$work/root/usr/share/postgresql/18/extension/"age* "$dest/share/extension/"
grep -q "^default_version = '1.8.0'$" "$dest/share/extension/age.control"
# The server runs as uid 999 and must read every file.
chmod -R a+rX "$dest"
find "$dest" -type f | sort
