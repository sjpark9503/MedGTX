#!/bin/sh
declare -A TPU_ID=(["1"]="10.26.120.162" ["2"]="10.45.120.210" ["3"]="10.66.23.242" ["4"]="10.67.73.42" ["5"]="10.119.205.34")

export TPU_IP_ADDRESS="${TPU_ID[$1]}"
export XRT_TPU_CONFIG="tpu_worker;0;$TPU_IP_ADDRESS:8470"
python run.py
