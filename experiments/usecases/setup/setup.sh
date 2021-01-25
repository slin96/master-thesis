#!/usr/bin/env bash

cd "$(dirname "$0")"

# create .env file based on .env-template
export $(cat .env | sed 's/#.*//g' | xargs)

docker-compose up -d

sleep 3
