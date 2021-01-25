#!/usr/bin/env bash

cd "$(dirname "$0")"

# install mmlib
pip install /shared/mmlib-0.0.1-py3-none-any.whl

python /node/experiments/usecases/u1/node.py --tmp_dir /shared --mongo_ip mongo-db --server_ip server-container --node_ip node-container > /shared/log-python-node.log