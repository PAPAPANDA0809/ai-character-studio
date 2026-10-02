#!/bin/bash
cd "$(dirname "$0")"
one() { dest="${1%%|*}"; url="${1#*|}"; for i in 1 2 3 4 5 6; do curl -sSfL --retry 5 -C - -o "$dest" "$url" < /dev/null && { echo "OK $dest"; return 0; }; echo "retry $i $dest"; sleep 5; done; echo "FAIL $dest"; }
export -f one
tr -d '\r' < downloads.txt | grep . | xargs -d '\n' -P 4 -I{} bash -c 'one "$@"' _ {}
echo ALLDONE
