#!/bin/sh
set -eu
# Root review/admission required. This script transfers files only, never launches.
PACKET=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
scp "$PACKET/notebook-capsule.tar" m0hawk@192.168.178.40:/home/m0hawk/.local/share/sepalith-campaign-20260915/cache-event-live-a-capsule.tar
ssh m0hawk@192.168.178.40 'test ! -e /home/m0hawk/.local/share/sepalith-campaign-20260915/cache-event-live-a-capsule && mkdir /home/m0hawk/.local/share/sepalith-campaign-20260915/cache-event-live-a-capsule && tar -xf /home/m0hawk/.local/share/sepalith-campaign-20260915/cache-event-live-a-capsule.tar -C /home/m0hawk/.local/share/sepalith-campaign-20260915/cache-event-live-a-capsule'
