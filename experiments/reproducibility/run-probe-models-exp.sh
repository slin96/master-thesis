#!/usr/bin/env bash

cd "$(dirname "$0")"

while getopts l: flag
do
    case "${flag}" in
        l) MMLIB=${OPTARG};;
    esac
done

# mandatory arguments
if [ ! "$MMLIB" ]; then
  echo "argument -l must be provided to specify the path for the mmlib .whl file"
  exit 1
fi

# make sure mmlib container is available
sh ../docker/mmlib/create-mmlib-container.sh -l $MMLIB

cd "$(dirname "$0")"

pwd

docker run --rm --name mmlib-probe -it -d mmlib
docker cp probe_models.py mmlib-probe:.
docker exec mmlib-probe mkdir /experiments
docker cp ../models mmlib-probe:/experiments
docker exec mmlib-probe python probe_models.py
docker kill mmlib-probe