#!/bin/bash
# Traza la posicion del choque en el tiempo a partir del muestreo axial.
# Uso:  choque.sh [directorio_del_caso]
CASE="${1:-.}"
cd "$CASE" || exit 1
D=postProcessing/axisSampling
[ -d "$D" ] || { echo "No hay $D. Â¿Corriste el caso?"; exit 1; }

printf "%8s %10s %10s %10s %12s\n" "t[s]" "Ma_max" "x(Ma_max)" "x_choque" "p_salida[Pa]"
for t in $(ls "$D" | sort -g); do
  F="$D/$t/axialLine.csv"
  [ -f "$F" ] || continue
  awk -F, -v T="$t" '
    NR==1 { for(i=1;i<=NF;i++){ gsub(/"/,"",$i)
              if($i=="x")c_x=i; if($i=="Ma")c_m=i; if($i=="p")c_p=i } ; next }
    {
      if($c_m > mx) { mx=$c_m; mxx=$c_x }        # pico de Mach
      if(prev != "") { d = prev-$c_m             # mayor caida = choque
                       if(d > bd) { bd=d; bx=$c_x } }
      prev=$c_m; lastp=$c_p
    }
    END { printf "%8s %10.4f %10.3f %10.3f %12.1f\n", T, mx, mxx, bx, lastp }
  ' "$F"
done
