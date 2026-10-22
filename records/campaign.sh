#!/usr/bin/env bash
# Keep searching for high free-order scores, recording every distinct play
# above 200 into records/above-200 (see records/above-200/INDEX.md).
#
#     records/campaign.sh [parallel searches] [minutes]     # default 4, 60
#
# Each search picks a random tower height and beam settings and a fresh
# seed; searches share the record directory safely. Logs go to runs/.
set -u
cd "$(dirname "$0")/.."
parallel=${1:-4}
minutes=${2:-60}
cargo build --release -q --manifest-path rust/Cargo.toml --bin tower || exit 1
bin=rust/target/release/tower
mkdir -p runs records/above-200
log=runs/campaign-$(date +%Y%m%d-%H%M%S).log
end=$(($(date +%s) + minutes * 60))
seed=$(date +%s)

pick() { local a=("$@"); echo "${a[RANDOM % ${#a[@]}]}"; }

search() {
    local seed=$1
    while [ "$(date +%s)" -lt "$end" ]; do
        local args=(
            --tops "$(pick 5 6 6 6 7)"
            --width "$(pick 100 150 200 300)"
            --per-design "$(pick 6 8 10 12)"
            --area-cost "$(pick 6 8 10 12 16)"
            --growth "$(pick 3 4 5)"
            --noise "$(pick 0 10 30 60)"
            --seed "$seed"
        )
        local left=$((end - $(date +%s)))
        [ "$left" -gt 0 ] || break
        echo "[$(date +%T)] start: ${args[*]}" >>"$log"
        timeout "$left" "$bin" "${args[@]}" --record-above 200 --record-dir records/above-200 \
            --out /dev/null 2>&1 | grep -E "recorded|new best" >>"$log"
        seed=$((seed + 1000))
    done
}

echo "campaign: $parallel searches for $minutes minutes, log $log"
for k in $(seq 1 "$parallel"); do
    search $((seed + k)) &
done
wait
echo "campaign finished; $(grep -c recorded "$log") new plays recorded"
head -12 records/above-200/INDEX.md
