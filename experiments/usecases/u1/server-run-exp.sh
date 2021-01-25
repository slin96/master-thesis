#!/usr/bin/env bash

while getopts m: flag
do
    case "${flag}" in
        m) MODEL=${OPTARG};;
    esac
done


echo "used model: $MODEL";

# mandatory arguments
if [ ! "$MODEL" ]; then
  echo "arguments -m must be provided to specify model"
  exit 1
fi

MODEL_NAME=$MODEL
MODEL_CODE="$MODEL.py"

echo "used MODEL: $MODEL_NAME";
echo "used MODEL: $MODEL_CODE";

cd "$(dirname "$0")"

# install mmlib
pip install /shared/mmlib-0.0.1-py3-none-any.whl

# wait for node to be ready for listening
sleep 3

python /server/experiments/usecases/u1/server.py --model $MODEL_NAME --tmp_dir /shared --model_code /server/experiments/models/$MODEL_CODE --import_root /server --mongo_ip mongo-db --server_ip server-container --node_ip node-container > /shared/log-server.log