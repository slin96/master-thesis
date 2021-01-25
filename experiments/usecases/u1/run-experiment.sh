#!/usr/bin/env bash

cd "$(dirname "$0")"

# TODO create shared folders

# start docker containers
sh ../setup/setup.sh

# run experiment code for node and server container
docker exec server-container sh /server/experiments/usecases/u1/server-run-exp.sh > server-log.log &
docker exec node-container sh /node/experiments/usecases/u1/node-run-exp.sh > node-log.log

# cleanup docker containers
sh ../setup/cleanup.sh
