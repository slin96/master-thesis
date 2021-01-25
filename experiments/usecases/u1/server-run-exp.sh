#!/usr/bin/env bash

cd "$(dirname "$0")"

# install mmlib
pip install /shared/mmlib-0.0.1-py3-none-any.whl

# wait for node to be ready for listening
sleep 3

python /server/experiments/usecases/u1/server.py --model resnet18 --tmp_dir /shared --model_code /server/experiments/models/resnet18.py --import_root /server --mongo_ip mongo-db --server_ip server-container --node_ip node-container > /shared/log-server.log