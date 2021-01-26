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

if test -f .env;
then

  export $(cat .env | sed 's/#.*//g' | xargs)


  # create shared folders
  echo "create mounted directories"
  mkdir -p $SHARED_MOUNTED_DIR
  mkdir -p $SHARED_MOUNTED_DIR/logs/baseline
  mkdir -p $SHARED_MOUNTED_DIR/logs/advanced1
  mkdir -p $SHARED_MOUNTED_DIR/logs/advanced2
  mkdir -p $SERVER_MOUNTED_DIR
  mkdir -p $NODE_MOUNTED_DIR
  echo "copy files to mounted directories"
  echo "create sub directories"
  mkdir -p $SERVER_MOUNTED_DIR/experiments
  mkdir -p $NODE_MOUNTED_DIR/experiments
  echo "copy to shared directory"
  cp $MMLIB $SHARED_MOUNTED_DIR
  echo "copy to node directory"
  cp -r ../../../usecases $NODE_MOUNTED_DIR/experiments
  echo "copy to server directory"
  cp -r ../../../usecases $SERVER_MOUNTED_DIR/experiments
  cp -r ../../../models $SERVER_MOUNTED_DIR/experiments


  docker-compose up -d

  sleep 3

else
  echo "BEFORE EXECUTION: create .env file based on .env-template"
fi
