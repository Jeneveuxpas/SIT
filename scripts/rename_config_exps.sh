#!/usr/bin/env bash
set -euo pipefail

ROOT="/workspace/SIT/exps"
MAP="CONFIG_RENAME_MAP.tsv"
APPLY=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --root) ROOT="$2"; shift 2 ;;
    --map) MAP="$2"; shift 2 ;;
    --apply) APPLY=true; shift ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

[[ -d "$ROOT" ]] || { echo "Experiment root does not exist: $ROOT" >&2; exit 1; }
[[ -f "$MAP" ]] || { echo "Mapping file does not exist: $MAP" >&2; exit 1; }

changed=0
missing=0
conflicts=0
while IFS=$'\t' read -r old new _; do
  [[ -z "$old" || "$old" == "old_path" ]] && continue
  old=${old%.yaml}; new=${new%.yaml}
  src="$ROOT/$old"; dst="$ROOT/$new"
  if [[ ! -d "$src" ]]; then ((missing+=1)); continue; fi
  if [[ -e "$dst" ]]; then echo "CONFLICT target exists: $new" >&2; ((conflicts+=1)); continue; fi
  if [[ "$APPLY" == true ]]; then mv -- "$src" "$dst"; echo "RENAMED: $old -> $new"; else echo "DRY RUN: $old -> $new"; fi
  ((changed+=1))
done < "$MAP"

echo "Summary: candidates=$changed missing=$missing conflicts=$conflicts mode=$([[ "$APPLY" == true ]] && echo apply || echo dry-run)"
[[ "$APPLY" == true ]] || echo "No directories were changed. Re-run with --apply after reviewing the list."
[[ "$conflicts" -eq 0 ]]
