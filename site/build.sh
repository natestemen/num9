#!/usr/bin/env bash
# Build the website into site/_site (or the directory given):
#
#     site/build.sh [output directory]
#
# Needs Rust with the wasm32-unknown-unknown target
# (`rustup target add wasm32-unknown-unknown`) and Python 3.
set -euo pipefail
cd "$(dirname "$0")/.."
out=${1:-site/_site}
manifest=rust/Cargo.toml

cargo build --release -q --manifest-path "$manifest" --target wasm32-unknown-unknown -p num9-web
target=$(cargo metadata --format-version 1 --no-deps --manifest-path "$manifest" |
    python3 -c 'import json, sys; print(json.load(sys.stdin)["target_directory"])')

rm -rf "$out"
mkdir -p "$out/js" "$out/data"
cp site/index.html site/style.css "$out/"
cp site/js/*.js "$out/js/"
cp "$target/wasm32-unknown-unknown/release/num9_web.wasm" "$out/num9.wasm"
cp site/challenges.json "$out/data/"
python3 site/build_data.py "$out"
touch "$out/.nojekyll"
echo "built $out ($(du -sh "$out" | cut -f1))"
