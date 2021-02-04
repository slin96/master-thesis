#!/usr/bin/env bash

while getopts a: flag
do
    case "${flag}" in
        a) APPROACH=${OPTARG};;
    esac
done
‚
# mandatory arguments
if [ ! "$APPROACH" ]; then
  echo "argument -a must be provided to specify the approach"
  exit 1
fi

cd "$(dirname "$0")"

python /node/experiments/workflows/basic/approaches/$APPROACH/node.py --tmp_dir /shared --mongo_ip mongo-db --server_ip server-container --node_ip node-container --log_dir /shared/logs/$APPROACH > /shared/logs/$APPROACH/python-node.log