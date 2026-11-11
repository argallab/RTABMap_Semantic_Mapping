#!/usr/bin/env bash
# Analysis 3 runs (inside gpu_map_8gb):  bash /app/analysis/tools/run_holdout.sh [db_name]
# 4 folds of tags (every 4th tag ID, sorted) x shift {0, -0.15 s}
# x tag variances {0.02/0.05, 0.2/999}, plus one
# no-holdout/no-shift validation run that must reproduce the exporter's poses.
# At most 3 runs at once (~1-2 GB each).
set -eo pipefail
DB_NAME="${1:-09-29-2.db}"
DB="/app/databases/${DB_NAME}"
BIN="/app/analysis/tools/build/tag_holdout"
OUT="/app/analysis/holdout/${DB_NAME%.db}"
mkdir -p "$OUT"

# Tag IDs from the DB's landmark links (to_id = -tag_id), sorted.
TAGS=$(python3 - "$DB" <<'EOF'
import sqlite3, sys
c = sqlite3.connect(f"file:{sys.argv[1]}?mode=ro", uri=True)
print(" ".join(str(t) for t in sorted({-r[0] for r in c.execute("select to_id from Link where type=8")})))
EOF
)
read -ra T <<< "$TAGS"
echo "tags (${#T[@]}): ${TAGS}"

CMDS=()
CMDS+=("$BIN $DB $OUT/validate")
for k in 0 1 2 3; do
  F=""
  for i in "${!T[@]}"; do (( i % 4 == k )) && F+="${T[$i]},"; done
  F="${F%,}"
  # current exporter-baseline tag variances (lin 0.02 / ang 0.05)
  CMDS+=("$BIN $DB $OUT/fold${k}_s0 --holdout $F")
  CMDS+=("$BIN $DB $OUT/fold${k}_s015 --holdout $F --shift -0.15")
  # live setting: lin 0.2 / ang 999 (position-only tags)
  CMDS+=("$BIN $DB $OUT/fold${k}_v2_s0 --holdout $F --lm-var-lin 0.2 --lm-var-ang 999")
  CMDS+=("$BIN $DB $OUT/fold${k}_v2_s015 --holdout $F --lm-var-lin 0.2 --lm-var-ang 999 --shift -0.15")
done
LOGS="$OUT/logs"; mkdir -p "$LOGS"
# Full stderr per run goes to $LOGS/<run>.log; the console shows progress and any error.
printf '%s\n' "${CMDS[@]}" | xargs -P 3 -I{} bash -c '
  cmd="{}"; name=$(basename $(echo $cmd | cut -d" " -f3))
  $cmd 2> "'"$LOGS"'/$name.log"; rc=$?
  grep -E "FAILED|FATAL|terminate|what\(\)|graph re-opt|icp round|done" "'"$LOGS"'/$name.log" | sed "s/^/[$name] /"
  [ $rc -ne 0 ] && echo "[$name] EXIT $rc (full log: '"$LOGS"'/$name.log)"; true'
echo "all runs finished -> $OUT"
