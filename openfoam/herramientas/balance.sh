#!/bin/bash
# Balance de masa entrada/salida en el ultimo tiempo. El test de estacionariedad.
# Uso:  balance.sh [directorio_del_caso]
CASE="${1:-.}"
cd "$CASE" || exit 1

IN=$(foamPostProcess -func "patchFlowRate(patch=inlet)"  -latestTime 2>/dev/null \
     | awk '/sum\(inlet\)/  {print $NF}' | tail -1)
OUT=$(foamPostProcess -func "patchFlowRate(patch=outlet)" -latestTime 2>/dev/null \
     | awk '/sum\(outlet\)/ {print $NF}' | tail -1)

awk -v i="$IN" -v o="$OUT" 'BEGIN {
  ai = (i<0 ? -i : i)
  printf "entrada  %10.4f kg/s\n", ai
  printf "salida   %10.4f kg/s\n", o
  if (ai > 0) {
    e = (o-ai)/ai*100; ae = (e<0 ? -e : e)
    printf "desbalance %8.2f %%   -> %s\n", e, \
      (ae < 2 ? "ESTACIONARIO" : (ae < 10 ? "casi, oscilacion residual" : "NO CONVERGIDO"))
  }
}'
