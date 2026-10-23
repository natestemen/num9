#!/usr/bin/env bash
# Build the website into site/_site (or the directory given):
#
#     site/build.sh [output directory]
#
# Needs Rust with the wasm32-unknown-unknown target
# (`rustup target add wasm32-unknown-unknown`) and Python 3. The puzzle
# tables take a minute or two to compute the first time and are cached in
# site/.cache.
set -euo pipefail
cd "$(dirname "$0")/.."
out=${1:-site/_site}
cache=site/.cache
manifest=rust/Cargo.toml

cargo build --release -q --manifest-path "$manifest" --target wasm32-unknown-unknown -p num9-web
cargo build --release -q --manifest-path "$manifest" --bin brute
target=$(cargo metadata --format-version 1 --no-deps --manifest-path "$manifest" |
    python3 -c 'import json, sys; print(json.load(sys.stdin)["target_directory"])')

# Positions solved in advance for the larger puzzle decks.
mkdir -p "$cache"
table() { # name, n, copies
    [ -s "$cache/$1.bin" ] || "$target/release/brute" "$2" "$3" --table "$cache/$1.bin" >&2
}
table one-0-4 5 1
table two-0-2 3 2

rm -rf "$out"
mkdir -p "$out/js" "$out/tables"
cp site/index.html site/style.css "$out/"
cp site/js/*.js "$out/js/"
cp "$target/wasm32-unknown-unknown/release/num9_web.wasm" "$out/num9.wasm"
cp "$cache"/*.bin "$out/tables/"
python3 site/build_data.py "$out"
touch "$out/.nojekyll"
echo "built $out ($(du -sh "$out" | cut -f1))"
