export TPU_IP_ADDRESS=10.12.96.50
export XRT_TPU_CONFIG="tpu_worker;0;$TPU_IP_ADDRESS:8470"
python run.py
