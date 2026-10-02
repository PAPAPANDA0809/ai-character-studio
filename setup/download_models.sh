#!/bin/bash
# Downloads every model in downloads.txt (dest|url), resuming partial files and retrying until complete.
# Usage: bash download_models.sh [list-file]
cd "$(dirname "$0")"
LIST="${1:-downloads.txt}"
one() {
  dest="${1%%|*}"; url="${1#*|}"
  size=$(curl -sIL "$url" | tr -d '\r' | awk 'tolower($1)=="content-length:"{v=$2} END{print v}')
  for i in $(seq 1 60); do
    have=$(stat -c %s "$dest" 2>/dev/null || echo 0)
    if [ -n "$size" ] && [ "$have" = "$size" ]; then echo "OK $dest"; return 0; fi
    curl -sSL --retry 3 -C - -o "$dest" "$url" < /dev/null
    sleep 3
  done
  echo "FAIL $dest"
}
export -f one
tr -d '\r' < "$LIST" | grep . | xargs -d '\n' -P 3 -I{} bash -c 'one "$@"' _ {}
echo ALLDONE
