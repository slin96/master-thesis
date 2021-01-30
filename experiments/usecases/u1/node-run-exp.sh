#!/usr/bin/env bash

while getopts a: flag
do
    case "${flag}" in
        a) APPROACH=${OPTARG};;
    esac
done


echo "used model: $MODEL";

# mandatory arguments
if [ ! "$APPROACH" ]; then
  echo "argument -a must be provided to specify the approach"
  exit 1
fi

cd "$(dirname "$0")"

python /node/experiments/usecases/u1/approaches/$APPROACH/node.py --tmp_dir /shared --mongo_ip mongo-db --server_ip server-container --node_ip node-container --log_dir /shared/logs/$APPROACH > /shared/logs/$APPROACH/python-node.log