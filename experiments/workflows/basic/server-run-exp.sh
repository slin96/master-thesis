#!/usr/bin/env bash

while getopts m:a: flag
do
    case "${flag}" in
        m) MODEL=${OPTARG};;
        a) APPROACH=${OPTARG};;
    esac
done


echo "used model: $MODEL";

# mandatory arguments
if [ ! "$MODEL" ] || [ ! "$APPROACH" ]; then
  echo "arguments -m and -a must be provided to specify model and approach"
  exit 1
fi

MODEL_NAME=$MODEL
MODEL_CODE="$MODEL.py"

cd "$(dirname "$0")"

# wait for node to be ready for listening
sleep 3

python /server/experiments/workflows/basic/approaches/$APPROACH/server.py --model $MODEL_NAME --tmp_dir /shared --model_code /server/experiments/models/$MODEL_CODE --import_root /server --mongo_ip mongo-db --server_ip server-container --node_ip node-container --log_dir /shared/logs/$APPROACH