#!/usr/bin/env bash
# Install apt packages on GitHub-hosted Ubuntu runners, surviving mirror outages.
#
# Usage: .github/scripts/apt-install.sh [apt-get install flags...] pkg...
#
# The runner's apt uses `mirror+file:/etc/apt/apt-mirrors.txt`, which points
# at azure.archive.ubuntu.com (fallback archive.ubuntu.com). Both live in the
# ubuntu.com DNS zone, so a Canonical DNS outage (e.g. 2026-11-11) or a slow
# Azure node stalls apt for the whole step timeout. We probe the primary mirror
# first; if it is unreachable, or apt fails/times out, we switch to independent
# mirrors outside ubuntu.com. Integrity is unchanged: apt still verifies every
# index and package against the Ubuntu archive keyring.
set -euo pipefail

MIRRORS_FILE=/etc/apt/apt-mirrors.txt
FALLBACK_MIRRORS=(
  "http://mirror.math.princeton.edu/pub/ubuntu/"
  "http://mirror.us.leaseweb.net/ubuntu/"
  "https://ubuntu.osuosl.org/ubuntu/"
)
APT_OPTS=(
  -o Acquire::Retries=3
  -o Acquire::http::Timeout=20
  -o Acquire::https::Timeout=20
  -o Acquire::ForceIPv4=true
)
UPDATE_TIMEOUT=${APT_UPDATE_TIMEOUT:-240}
INSTALL_TIMEOUT=${APT_INSTALL_TIMEOUT:-600}

# shellcheck source=/dev/null
codename=$(. /etc/os-release && echo "${VERSION_CODENAME}")
switched=0

primary=""
if [[ -f "$MIRRORS_FILE" ]]; then
  primary=$(grep -m1 -oE '^https?://[^[:space:]]+' "$MIRRORS_FILE" || true)
fi
primary=${primary:-http://azure.archive.ubuntu.com/ubuntu/}

probe() {
  curl -fsS -4 --max-time 10 -o /dev/null "${1%/}/dists/${codename}/InRelease"
}

use_fallback() {
  [[ $switched -eq 1 ]] && return 0
  switched=1
  echo "::warning title=apt mirror fallback::$1; switching to independent Ubuntu mirrors"
  local old=""
  [[ -f "$MIRRORS_FILE" ]] && old=$(grep -oE '^https?://[^[:space:]]+' "$MIRRORS_FILE" || true)
  {
    printf '%s\n' "${FALLBACK_MIRRORS[@]}"
    if [[ -n "$old" ]]; then printf '%s\n' "$old"; fi
  } | sudo tee "$MIRRORS_FILE" >/dev/null
  # Sources that bypass the mirror list (e.g. security.ubuntu.com) also move.
  sudo sed -i -E \
    "s#https?://(azure\.archive|archive|security)\.ubuntu\.com/ubuntu/?#${FALLBACK_MIRRORS[0]}#g" \
    /etc/apt/sources.list /etc/apt/sources.list.d/*.sources /etc/apt/sources.list.d/*.list \
    2>/dev/null || true
  echo "apt mirrors now:"; cat "$MIRRORS_FILE"
}

apt_update() { sudo timeout "$UPDATE_TIMEOUT" apt-get update "${APT_OPTS[@]}"; }
apt_install() {
  sudo DEBIAN_FRONTEND=noninteractive timeout "$INSTALL_TIMEOUT" \
    apt-get install -y "${APT_OPTS[@]}" "$@"
}

echo "primary apt mirror: $primary (codename $codename)"
if ! probe "$primary"; then
  use_fallback "primary mirror $primary is unreachable"
fi

if ! apt_update; then
  if [[ $switched -eq 1 ]]; then exit 1; fi
  use_fallback "apt-get update failed or timed out"
  apt_update
fi

if ! apt_install "$@"; then
  if [[ $switched -eq 1 ]]; then exit 1; fi
  use_fallback "apt-get install failed or timed out"
  apt_update
  apt_install "$@"
fi
