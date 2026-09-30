#!/bin/bash
# Runs in WSL. Finishes the watchdog's authorized deadline delete of dropbear-train once `brev login` has been redone:
# polls non-interactively every 2 min for up to 24 h; when auth works, deletes the instance, confirms it is gone and
# appends {"event": "deleted"} to logs/brev/watchdog.log. Log: logs/brev/delete_retry.log
BREV=$HOME/.local/bin/brev; INST=dropbear-train
REPO=$(cd "$(dirname "$0")/../.." && pwd)
LOG=$REPO/logs/brev/delete_retry.log; WD=$REPO/logs/brev/watchdog.log
for i in $(seq 1 720); do
  if ls_out=$($BREV ls --no-check-latest < /dev/null 2>&1); then
    if ! echo "$ls_out" | grep -q "$INST"; then
      echo "$(date -Is) $INST not listed: already deleted" >> $LOG
      echo "{\"time\": \"$(date -Is)\", \"event\": \"deleted\", \"by\": \"delete_when_authed.sh (already gone)\"}" >> $WD; exit 0
    fi
    echo "$(date -Is) auth OK, deleting $INST" >> $LOG
    $BREV delete $INST --no-check-latest < /dev/null >> $LOG 2>&1
    sleep 60
    if ! $BREV ls --no-check-latest < /dev/null 2>&1 | grep -q "$INST"; then
      echo "{\"time\": \"$(date -Is)\", \"event\": \"deleted\", \"by\": \"delete_when_authed.sh\"}" >> $WD
      echo "$(date -Is) deleted" >> $LOG; exit 0
    fi
  else
    [ $((i % 15)) -eq 1 ] && echo "$(date -Is) waiting for brev login (auth fails non-interactively)" >> $LOG
  fi
  sleep 120
done
echo "$(date -Is) gave up after 24 h" >> $LOG
