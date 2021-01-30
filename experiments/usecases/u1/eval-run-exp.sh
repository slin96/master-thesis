#!/usr/bin/env bash

while getopts a: flag
do
    case "${flag}" in
        a) APPROACH=${OPTARG};;
    esac
done

# mandatory arguments
if [ ! "$APPROACH" ]; then
  echo "argument -a must be provided to specify the approach"
  exit 1
fi

cd "$(dirname "$0")"

python /eval/experiments/usecases/u1/approaches/$APPROACH/eval.py --tmp_dir /shared --log_dir /shared/logs/$APPROACH > /shared/logs/$APPROACH/python-eval.log