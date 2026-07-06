#!/usr/bin/env sh
set -eu

BUILD_TYPE="${BUILD_TYPE:-release}"
CEP_OUTPUT_DIR="${CEP_OUTPUT_DIR:-.dist/cep}"

rm -rf build
meson setup build -Dbuildtype="${BUILD_TYPE}"
meson compile -C build stelline_cep

mkdir -p "${CEP_OUTPUT_DIR}"
cp build/stelline.cep "${CEP_OUTPUT_DIR}/stelline.cep"
