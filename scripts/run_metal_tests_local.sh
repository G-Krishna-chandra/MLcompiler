#!/usr/bin/env bash
set -euo pipefail

echo "== host info =="
uname -a
sw_vers
echo "user: $(whoami) uid: $(id -u)"

echo "== configure/build (Metal required) =="
cmake -S . -B build -DMLC_REQUIRE_METAL=ON
cmake --build build

echo "== diagnostic =="
./build/bin/metal_diag

echo "== metal tests =="
cd build
ctest -V -R "MetalRuntimeTest|MetalAttention|metal_diag" --output-on-failure
