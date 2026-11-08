#!/usr/bin/env bash
# Prove that the pinned PGDG package is built from the ASF 1.8.0 release.
#
# PGDG names the version 1.8.0~rc0 because it builds from the GitHub tag
# PG18/v1.8.0-rc0. homelab#332 requires that the pin matches the ASF release
# before it is used. Checked 2026-11-08: the two source trees are identical
# (381 files), and the one Debian patch changes only the PGOPTIONS of the
# package's own regression tests in the Makefile.
set -euo pipefail

here=$(cd "$(dirname "$0")" && pwd)
# shellcheck source=pins.env
source "$here/pins.env"

work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
mkdir "$work/asf" "$work/pgdg" "$work/debian"

curl -fsSL -o "$work/asf.tar.gz" "$AGE_ASF_URL"
echo "$AGE_ASF_SHA512  $work/asf.tar.gz" | sha512sum -c -
curl -fsSL -o "$work/orig.tar.gz" "$AGE_ORIG_URL"
echo "$AGE_ORIG_SHA256  $work/orig.tar.gz" | sha256sum -c -
curl -fsSL -o "$work/debian.tar.xz" "$AGE_DEBIAN_URL"

tar -xzf "$work/asf.tar.gz" -C "$work/asf"
tar -xzf "$work/orig.tar.gz" -C "$work/pgdg"
tar -xJf "$work/debian.tar.xz" -C "$work/debian"

diff -r "$work/asf/apache-age-1.8.0" "$work/pgdg/age-PG18-v1.8.0-rc0"
echo "PGDG source tree == ASF apache-age-1.8.0-src.tar.gz"

patches=$(cat "$work/debian/debian/patches/series")
if [ "$patches" != "extra-instance-args" ]; then
  echo "unexpected Debian patches: $patches" >&2
  exit 1
fi
touched=$(grep '^+++ ' "$work/debian/debian/patches/extra-instance-args" | awk '{print $2}')
if [ "$touched" != "b/Makefile" ]; then
  echo "the Debian patch touches more than the Makefile: $touched" >&2
  exit 1
fi
echo "Debian patches: extra-instance-args (Makefile only)"
