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

# copy mm lib to local directory, is used in Dockerfile
cp $MMLIB .

# build docker container mmlib
docker build -t mmlib .