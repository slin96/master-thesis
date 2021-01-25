#!/usr/bin/env bash

cd "$(dirname "$0")"

echo "setup mounted volumes and start docker containers"
sh ./setup/setup.sh

echo "run experiment code for node and server container"
docker exec server-container sh /server/experiments/usecases/u1/server-run-exp.sh > server-log.log &
docker exec node-container sh /node/experiments/usecases/u1/node-run-exp.sh > node-log.log

echo "experiments done - cleanup docker containers"
sh ./setup/cleanup.sh


