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

# copy mmlib to local directory, is used in Dockerfile
# remove old version if exists
rm mmlib-0.0.1-py3-none-any.whl
cp $MMLIB .

# build docker container mmlib
docker build -t mmlib .