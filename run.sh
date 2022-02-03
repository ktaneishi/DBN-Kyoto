#!/bin/bash

if [ ! -d theano ]; then
    python3 -m venv theano
    source theano/bin/activate
    pip install theano
else
    source theano/bin/activate
fi

python main.py
