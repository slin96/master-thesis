#!/usr/bin/env bash

while getopts m: flag
do
    case "${flag}" in
        m) MODEL=${OPTARG};;
    esac
done

# mandatory arguments
if [ ! "$MODEL" ]; then
  echo "arguments -m must be provided to specify model"
  echo "valid models are: mobilenet, googlenet, resnet18, resnet50, resnet152"
  exit 1
fi

cd "$(dirname "$0")"

echo "setup mounted volumes and start docker containers"
sh ./setup/setup.sh

echo "run experiment code for node and server container"
docker exec server-container sh /server/experiments/usecases/u1/server-run-exp.sh -m $MODEL > server-log.log &
docker exec node-container sh /node/experiments/usecases/u1/node-run-exp.sh > node-log.log

echo "experiments done - cleanup docker containers"
sh ./setup/cleanup.sh


