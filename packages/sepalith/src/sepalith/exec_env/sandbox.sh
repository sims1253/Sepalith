#!/bin/sh
# Sandbox launcher for one R execution worker. Run as:
#   prlimit --as=BYTES unshare -rnm --propagation private sh sandbox.sh WORKDIR MB -- CMD...
# Inside the new user + network + mount namespace: mount a private tmpfs of MB
# megabytes at WORKDIR (the only writable path), remount every other mount
# read-only, then exec CMD with HOME, TMPDIR and the working directory on the tmpfs.
# The network namespace has only a downed loopback, so no network is reachable.
set -eu
work="$1"; size="$2"; shift 3
mount -t tmpfs -o "size=${size}m,mode=700" tmpfs "$work"
for m in $(awk '{print $2}' /proc/self/mounts | sort -u); do
  [ "$m" = "$work" ] && continue
  mount -o remount,bind,ro "$m" 2>/dev/null || true
done
cd "$work"
export HOME="$work" TMPDIR="$work" R_USER="$work" R_LIBS_USER="$work/none" LANG=C.UTF-8 LC_ALL=C.UTF-8
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY no_proxy
exec "$@"
