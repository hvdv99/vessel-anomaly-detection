#!/bin/sh

"${SHELL}" <(curl -L micro.mamba.pm/install.sh)
micromamba self-update
micromamba env create -f environment.yml
micromamba activate vessel-anomaly-detection
