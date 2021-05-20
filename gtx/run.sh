#!/bin/sh
declare -A TPU_ID=(["1"]="10.12.96.50" ["2"]="10.110.121.146" ["3"]="10.105.114.34" ["4"]="10.77.132.226" ["5"]="10.120.177.122")

export TPU_IP_ADDRESS="${TPU_ID[$1]}"
export XRT_TPU_CONFIG="tpu_worker;0;$TPU_IP_ADDRESS:8470"
python run.py
