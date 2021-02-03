#!/usr/bin/env bash

while getopts m:a:l: flag
do
    case "${flag}" in
        m) MODEL=${OPTARG};;
        a) APPROACH=${OPTARG};;
        l) MMLIB=${OPTARG};;
    esac
done

# mandatory arguments
if [ ! "$MODEL" ]; then
  echo "argument -m must be provided to specify model"
  echo "valid models are: mobilenet, googlenet, resnet18, resnet50, resnet152"
  exit 1
fi

if [ ! "$APPROACH" ]; then
  echo "argument -a must be provided to specify the approach"
  echo "valid approaches are: baseline, advanced1, advanced2"
  exit 1
fi

if [ ! "$MMLIB" ]; then
  echo "argument -l must be provided to specify the path for the mmlib .whl file"
  exit 1
fi

cd "$(dirname "$0")"

echo "setup mounted volumes and start docker containers"
sh ./setup/setup.sh -l $MMLIB

echo "run experiment code for node and server container"
docker exec server-container sh /server/experiments/workflows/basic/server-run-exp.sh -m $MODEL -a $APPROACH > server-log.log &
docker exec node-container sh /node/experiments/workflows/basic/node-run-exp.sh -a $APPROACH > node-log.log

echo "eval results"
docker exec eval-container sh /eval/experiments/workflows/basic/eval-run-exp.sh -a $APPROACH > eval-log.log

echo "experiments done - cleanup docker containers"
sh ./setup/cleanup.sh


