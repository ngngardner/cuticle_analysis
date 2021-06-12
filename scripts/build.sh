#!/bin/bash

echo "create venv..."
python3 -m venv .env

activate () {
    . .env/bin/activate
}

echo "install dependencies..."
activate
pip install wheel
pip install -r requirements.txt
pip install -e .

echo "extracting dataset..."
[ ! -d "./dataset" ] && python -c '
import sys
from zipfile import PyZipFile
zf = PyZipFile("dataset.zip")
zf.extractall(path="./dataset")
' || echo "Directory dataset exists, skipping extract..."

echo "Done."
