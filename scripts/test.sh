#!/bin/bash

activate () {
    . .env/bin/activate
}

activate

python tests/test_dataset.py
