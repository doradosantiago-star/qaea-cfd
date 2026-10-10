#!/bin/bash
# Perfil de Mach A TRAVES de la seccion: revela si el flujo es cuasi-1D.
# Uso:  transversal.sh <caso> <x> <semialtura_en_esa_x>
CASE="$1"; X="$2"; H="$3"
[ -z "$H" ] && { echo "uso: transversal.sh <caso> <x> <semialtura>"; exit 1; }
cd "$CASE" || exit 1

cat > system/xsec <<EOF
type            sets;
libs            ("libsampling.so");
interpolationScheme cellPoint;
setFormat       csv;
fields          (Ma p U);
sets
(
    corte { type lineUniform; axis y; start ($X 0.0 0.05); end ($X $H 0.05); nPoints 25; }
);
EOF

foamPostProcess -func xsec -latestTime > /dev/null 2>&1
F=$(find postProcessing/xsec -name "corte.csv" | sort | tail -1)
[ -f "$F" ] || { echo "el muestreo no produjo datos"; exit 1; }

echo "Mach a traves de la seccion en x = $X   (y=0 eje, y=$H pared)"
awk -F, -v H="$H" '
  NR==1 { for(i=1;i<=NF;i++){ gsub(/"/,"",$i); if($i=="y")c_y=i; if($i=="Ma")c_m=i }; next }
  { printf "  y/h=%5.3f   Ma=%7.4f\n", $c_y/H, $c_m
    if($c_m>mx) mx=$c_m ; if(mn=="" || $c_m<mn) mn=$c_m }
  END { printf "\n  variacion %.1f %%  -> %s\n", (mx-mn)/mx*100,
          ((mx-mn)/mx < 0.05 ? "UNIFORME (cuasi-1D valido)" : "NO UNIFORME (hay chorro/capa de corte)") }
' "$F"
