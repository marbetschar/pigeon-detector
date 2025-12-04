#!/bin/bash
# Run this script on Ubuntu 22.04 x64:
# apt update
# apt upgrade
# reboot

apt update
apt install python3 python-is-python3 build-essentials graphviz libgraphviz-dev

pip install hailo_dataflow_compiler-3.33.0-py3-none-linux_x86_64.whl
hailo -h
