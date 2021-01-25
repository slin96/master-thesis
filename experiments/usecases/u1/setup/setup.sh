#!/usr/bin/env bash

cd "$(dirname "$0")"

if test -f .env;
then

  export $(cat .env | sed 's/#.*//g' | xargs)


  # create shared folders
  echo "create mounted directories"
  mkdir -p $SHARED_MOUNTED_DIR
  mkdir -p $SERVER_MOUNTED_DIR
  mkdir -p NODE_MOUNTED_DIR
  echo "copy files to mounted directories"
  echo "create sub directories"
  mkdir -p $SERVER_MOUNTED_DIR/experiments
  mkdir -p $NODE_MOUNTED_DIR/experiments
  echo "copy to shared directory"
  # TODO automatically build mmlib
  # TODO copy mmlib to shared
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
