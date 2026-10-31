#!/usr/bin/env bash
# Pin the campaign's R interpreter with rig, and optionally scaffold rv.
#
# Why this exists: the C18b GRPO pilot refused to resume on 2026-10-31 because
# a pacman upgrade moved /usr/bin/Rscript from R 4.6.1 to 4.6.2 while the
# reward buffers were verified by the R 4.6.1 parser
# (sepalith.training.rl.reward_v21.RewardV21Error). The durable fix is to stop
# using the distro's R at all:
#
#   rig  installs R into /opt/R/<version> and leaves versioned links
#        (R-4.6.1 / Rscript-4.6.1), so pacman cannot move it
#   rv   records the R version and the package set declaratively (optional)
#
# Usage:
#   scripts/env/install_r_rig.sh              install pinned rig + R 4.6.1
#   scripts/env/install_r_rig.sh --check      verify only; write nothing
#   scripts/env/install_r_rig.sh --with-rv    also install pinned rv and
#                                             scaffold an rv project
#   scripts/env/install_r_rig.sh --rv-dir D   rv project directory
#                                             (default ~/.local/share/sepalith/r-env)
#   scripts/env/install_r_rig.sh --help
#
# Run as your normal user, NOT with sudo. Only two writes leave $HOME:
# /opt/R/4.6.1 and the /usr/local/bin quick links, and the script runs
# `sudo rig add --admin --no-default 4.6.1` for those. Everything else goes
# under ~/.local. R is added with --no-default on purpose: the system
# R/Rscript commands keep pointing at /usr/bin, so already-bound recipes stay
# on their pinned interpreter until they are deliberately rebound.
#
# This script is idempotent: an existing matching rig, R 4.6.1, rv and
# rproject.toml are kept.
set -euo pipefail

R_VERSION=4.6.1
RIG_VERSION=0.11.1
RV_VERSION=0.23.3

R_PREFIX=/opt/R/$R_VERSION
RSCRIPT=$R_PREFIX/bin/Rscript
LOCAL_BIN=$HOME/.local/bin
RV_DIR=$HOME/.local/share/sepalith/r-env

MODE=install
WITH_RV=0
TMP=

usage() {
  cat <<'EOF'
Usage: scripts/env/install_r_rig.sh [--check] [--with-rv] [--rv-dir DIR]

  --check        report the pinned rig, R and rv state; write nothing
  --with-rv      also install pinned rv and scaffold an rv project
  --rv-dir DIR   rv project directory (default ~/.local/share/sepalith/r-env)
  -h, --help     show this help

Installs rig 0.11.1 under ~/.local and R 4.6.1 under /opt/R/4.6.1. The R
install and its /usr/local/bin quick links need sudo (admin mode). Run this
script as your normal user; it calls sudo itself.
EOF
}

say() { printf '%s\n' "$*"; }
ok() { printf 'ok: %s\n' "$*"; }
warn() { printf 'warn: %s\n' "$*" >&2; }
die() {
  printf 'error: %s\n' "$*" >&2
  exit 1
}

cleanup() {
  if [[ -n $TMP && -d $TMP ]]; then
    rm -rf -- "$TMP"
  fi
}

case $(uname -m) in
  x86_64 | amd64)
    RIG_ARCH=x86_64
    RV_ARCH=x86_64
    RIG_SHA=757b5b7886b09a8a97f35ab6e71546b68044d323ffbcaa68adcbe818409af156
    RV_SHA=749ffd6cfd78b10834e11074f62e347e7d214b50ee1356de557733633aa5cf87
    ;;
  aarch64 | arm64)
    RIG_ARCH=aarch64
    RV_ARCH=aarch64
    RIG_SHA=2b796885bc0398ae7700f8c95847a855e29c359769f31e1993569274316fb508
    RV_SHA=75d652c42dd5ea7c9c600b9a9f251442e22ac0ee7b1c8528b32227ab20c2552a
    ;;
  *)
    die "unsupported architecture $(uname -m); this script pins x86_64 and aarch64 builds"
    ;;
esac

RIG_URL=https://github.com/r-lib/rig/releases/download/v$RIG_VERSION/rig-linux-$RIG_ARCH-$RIG_VERSION.tar.gz
RV_URL=https://github.com/A2-ai/rv/releases/download/v$RV_VERSION/rv-v$RV_VERSION-$RV_ARCH-unknown-linux-gnu.tar.gz

while (($#)); do
  case $1 in
    --check)
      MODE=check
      ;;
    --with-rv)
      WITH_RV=1
      ;;
    --rv-dir)
      (($# >= 2)) || die "--rv-dir needs a directory"
      RV_DIR=$2
      shift
      ;;
    -h | --help)
      usage
      exit 0
      ;;
    *)
      usage >&2
      die "unknown argument: $1"
      ;;
  esac
  shift
done

fetch() {
  local url=$1 dest=$2 sha=$3
  say "downloading $url"
  if command -v curl >/dev/null 2>&1; then
    curl -fsSL --retry 3 --retry-delay 2 -o "$dest" "$url"
  elif command -v wget >/dev/null 2>&1; then
    wget -q -O "$dest" "$url"
  else
    die "need curl or wget to download $url"
  fi
  printf '%s  %s\n' "$sha" "$dest" | sha256sum -c --status - ||
    die "checksum mismatch for $url"
}

rig_bin() {
  if [[ -x $LOCAL_BIN/rig ]]; then
    printf '%s\n' "$LOCAL_BIN/rig"
  elif command -v rig >/dev/null 2>&1; then
    command -v rig
  else
    return 1
  fi
}

rig_matches() {
  local bin out
  bin=$(rig_bin) || return 1
  out=$("$bin" --version 2>/dev/null) || return 1
  [[ $out == *"$RIG_VERSION"* ]]
}

r_ok() {
  [[ -x $RSCRIPT ]] || return 1
  "$RSCRIPT" --version 2>/dev/null | grep -Fq " $R_VERSION "
}

rv_bin() {
  if [[ -x $LOCAL_BIN/rv ]]; then
    printf '%s\n' "$LOCAL_BIN/rv"
  elif command -v rv >/dev/null 2>&1; then
    command -v rv
  else
    return 1
  fi
}

rv_matches() {
  local bin out
  bin=$(rv_bin) || return 1
  out=$("$bin" --version 2>/dev/null) || return 1
  [[ $out == *"$RV_VERSION"* ]]
}

install_rig() {
  local tarball
  if rig_matches; then
    ok "rig $RIG_VERSION already installed ($(rig_bin))"
    return 0
  fi
  mkdir -p "$LOCAL_BIN"
  tarball=$TMP/rig.tar.gz
  fetch "$RIG_URL" "$tarball" "$RIG_SHA"
  tar -xzf "$tarball" -C "$HOME/.local"
  [[ -x $LOCAL_BIN/rig ]] || die "rig binary missing after unpacking $RIG_URL"
  ok "installed $("$LOCAL_BIN/rig" --version) at $LOCAL_BIN/rig"
}

install_r() {
  local rig
  if r_ok; then
    ok "R $R_VERSION already installed at $R_PREFIX"
    return 0
  fi
  rig=$(rig_bin) || die "rig is not installed; re-run without --check first"
  say "rig add --admin --no-default $R_VERSION installs to $R_PREFIX"
  say "Arch is not a rig-supported Linux distro, so rig uses its portable build."
  say "This write needs sudo; the script is running it for you."
  if [[ $(id -u) -eq 0 ]]; then
    "$rig" add --admin --no-default "$R_VERSION"
  else
    command -v sudo >/dev/null 2>&1 ||
      die "sudo is required to write $R_PREFIX; run as an administrator"
    sudo "$rig" add --admin --no-default "$R_VERSION"
  fi
  r_ok || die "rig finished but $RSCRIPT is not R $R_VERSION"
  ok "installed R $R_VERSION at $R_PREFIX"
}

check_links() {
  local rc=0 stem link
  for stem in R Rscript; do
    link=/usr/local/bin/$stem-$R_VERSION
    if [[ -x $link ]]; then
      ok "versioned link $link"
    elif [[ -x $LOCAL_BIN/$stem-$R_VERSION ]]; then
      ok "versioned link $LOCAL_BIN/$stem-$R_VERSION"
    else
      warn "no versioned link for $stem (expected /usr/local/bin/$stem-$R_VERSION)"
      rc=1
    fi
  done
  return "$rc"
}

ensure_links() {
  local stem target link
  for stem in R Rscript; do
    target=$R_PREFIX/bin/$stem
    link=/usr/local/bin/$stem-$R_VERSION
    if [[ -x $link ]]; then
      ok "versioned link $link"
      continue
    fi
    mkdir -p "$LOCAL_BIN"
    ln -sfn "$target" "$LOCAL_BIN/$stem-$R_VERSION"
    ok "versioned link $LOCAL_BIN/$stem-$R_VERSION -> $target"
  done
}

install_rv() {
  local tarball src
  if rv_matches; then
    ok "rv $RV_VERSION already installed ($(rv_bin))"
    return 0
  fi
  mkdir -p "$LOCAL_BIN" "$TMP/rv"
  tarball=$TMP/rv.tar.gz
  fetch "$RV_URL" "$tarball" "$RV_SHA"
  tar -xzf "$tarball" -C "$TMP/rv"
  src=$TMP/rv/rv
  if [[ ! -f $src ]]; then
    src=$(find "$TMP/rv" -type f -name rv -print -quit)
  fi
  [[ -n $src && -f $src ]] || die "rv binary missing after unpacking $RV_URL"
  install -m 0755 "$src" "$LOCAL_BIN/rv"
  ok "installed $("$LOCAL_BIN/rv" --version) at $LOCAL_BIN/rv"
}

write_rv_project() {
  local dir=$1
  mkdir -p "$dir"
  if [[ -e $dir/rproject.toml ]]; then
    ok "kept existing $dir/rproject.toml"
    return 0
  fi
  cat >"$dir/rproject.toml" <<'EOF'
# rv project for the Sepalith reward / exec_env R stack.
#
# The exact patch version is deliberate: RewardV21Scorer compares the
# reward-buffer parser identity ("R 4.6.1 (2026-06-24)") with the probe's
# identity, so a 4.6.x patch bump is not interchangeable.
[project]
name = "sepalith-r"
r_version = "4.6.1"

repositories = [
    { alias = "PPM", url = "https://packagemanager.posit.co/cran/latest" },
]

# Empty on purpose. The R code in packages/sepalith/src/sepalith uses only
# base and recommended packages that ship with R itself:
#   codetools (r_pool.R), stats/utils/methods/tools/parallel (r_worker.R),
# and base::parse (reward_parse_only.R). There are no CRAN dependencies to
# lock today; add them here when the reward / exec_env stack gains one.
dependencies = []
EOF
  ok "wrote $dir/rproject.toml (r_version = $R_VERSION, no external packages)"
}

check_state() {
  local rc=0
  if rig_matches; then
    ok "rig: $("$(rig_bin)" --version) at $(rig_bin)"
  else
    warn "rig $RIG_VERSION is not installed under $LOCAL_BIN or on PATH"
    rc=1
  fi
  if r_ok; then
    ok "R: $("$RSCRIPT" --version 2>/dev/null | head -n1) at $RSCRIPT"
  else
    warn "R $R_VERSION is not installed at $RSCRIPT"
    rc=1
  fi
  check_links || rc=1
  if ((WITH_RV)); then
    if rv_matches; then
      ok "rv: $("$(rv_bin)" --version) at $(rv_bin)"
    else
      warn "rv $RV_VERSION is not installed under $LOCAL_BIN or on PATH"
      rc=1
    fi
    if [[ -f $RV_DIR/rproject.toml ]]; then
      ok "rv project: $RV_DIR/rproject.toml"
    else
      warn "rv project missing: $RV_DIR/rproject.toml"
      rc=1
    fi
  fi
  say "future recipe binding: runtime.reward.rscript = $RSCRIPT"
  say "do not rebind C18/C18b; see docs/training/plan-20260927/R-ENVIRONMENT.md"
  return "$rc"
}

if [[ $MODE == check ]]; then
  if check_state; then
    exit 0
  else
    exit 1
  fi
fi

TMP=$(mktemp -d -t sepalith-rig-XXXXXX)
trap cleanup EXIT

install_rig
install_r
ensure_links
if ((WITH_RV)); then
  install_rv
  write_rv_project "$RV_DIR"
fi

say ""
say "pinned interpreter: $RSCRIPT"
say "identity:           $("$RSCRIPT" --version | head -n1)"
say "bind future recipes: \"rscript\": \"$RSCRIPT\""
say "already-bound C18/C18b recipes must not be rebound mid-run."
