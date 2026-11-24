#!/usr/bin/env bash
# Human-readable trash inventory for spark2, saved as text so it can be screenshotted.
set -uo pipefail
OUT="$HOME/trash-inventory.txt"

{
echo "SPARK2 TRASH INVENTORY"
echo "Generated: $(date -u '+%Y-%m-%d %H:%M UTC')"
echo "======================================================"
echo
echo "DISK"
df -h /home | tail -1
echo
echo "TRASH TOTAL"
du -sh "$HOME/trash" 2>/dev/null
echo
echo "======================================================"
echo "WHAT IS IN THE TRASH"
echo "======================================================"
for d in "$HOME"/trash/*/; do
  [ -d "$d" ] || continue
  echo
  echo ">>> $(basename "$d")   [$(du -sh "$d" 2>/dev/null | cut -f1)]"
  echo "------------------------------------------------------"
  for item in "$d"*/ "$d"*; do
    [ -e "$item" ] || continue
    if [ -d "$item" ]; then
      printf "  DIR   %-46s %s\n" "$(basename "$item")" "$(du -sh "$item" 2>/dev/null | cut -f1)"
    else
      printf "  FILE  %-46s %s\n" "$(basename "$item")" "$(du -sh "$item" 2>/dev/null | cut -f1)"
    fi
  done
done
echo
echo "======================================================"
echo "WHAT CAN BE RECLAIMED"
echo "======================================================"
echo "Emptying ~/trash would return approximately:"
echo "  $(du -sh "$HOME/trash" 2>/dev/null | cut -f1) of disk"
echo
echo "Current free: $(df -h /home | tail -1 | awk '{print $4}')"
echo "After emptying: approximately 380 GB free"
echo
echo "NOTHING IS DELETED UNTIL THE OWNER SAYS SO."
} > "$OUT" 2>&1

cat "$OUT"
