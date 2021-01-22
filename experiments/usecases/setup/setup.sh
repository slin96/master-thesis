#!/usr/bin/env bash

cd "$(dirname "$0")"

export $(cat .env | sed 's/#.*//g' | xargs)

docker-compose up -d

sleep 3
