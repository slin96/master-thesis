#!/bin/bash

# operate in the current dir
cd "$(dirname "$0")"

VENV=../../venv
REQ=../../../requirements.txt
LIB=../../../../mmlib

# build mmlib library
sh $LIB/generate-archives.sh

python3 -m venv $VENV
source $VENV/bin/activate
python3 -m pip install --upgrade pip

# install mmlib library
pip install $LIB/dist/mmlib-0.0.1-py3-none-any.whl

# install other requirements
pip install -r $REQ


