#!/usr/bin/env bash

cd "$(dirname "$0")" # set working dir to script dir
source .venv/bin/activate
python3 app.py "$@"