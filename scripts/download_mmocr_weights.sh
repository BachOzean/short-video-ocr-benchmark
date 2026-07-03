#!/usr/bin/env bash
set -euo pipefail

out_dir="${1:-./weights/mmocr}"
mkdir -p "$out_dir"

download() {
  local url="$1"
  local out="$2"
  if [[ -s "$out" ]]; then
    echo "exists: $out"
    return 0
  fi
  curl -L --fail --retry 4 -o "$out" "$url"
}

download \
  'https://download.openmmlab.com/mmocr/textrecog/sar/sar_r31_parallel_decoder_chineseocr_20210507-b4be8214.pth' \
  "$out_dir/sar_r31_parallel_decoder_chineseocr_20210507-b4be8214.pth"

download \
  'https://download.openmmlab.com/mmocr/textdet/dbnetpp/dbnetpp_resnet50_fpnc_1200e_icdar2015/dbnetpp_resnet50_fpnc_1200e_icdar2015_20221025_185550-013730aa.pth' \
  "$out_dir/dbnetpp_resnet50_fpnc_1200e_icdar2015_20221025_185550-013730aa.pth"

download \
  'https://download.openmmlab.com/mmocr/textdet/panet/panet_resnet18_fpem-ffm_600e_icdar2015/panet_resnet18_fpem-ffm_600e_icdar2015_20220826_144817-be2acdb4.pth' \
  "$out_dir/panet_resnet18_fpem-ffm_600e_icdar2015_20220826_144817-be2acdb4.pth"
