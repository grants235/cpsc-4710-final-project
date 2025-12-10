#!/usr/bin/env bash

# Assemble checkpoint
cat best-epoch=17.ckpt.part.* > best-epoch=17.ckpt

# Download and unzip dataset
wget https://s3.amazonaws.com/fast-ai-imageclas/CUB_200_2011.tgz
tar -xvzf CUB_200_2011.tgz

# Install requirements
pip install -r requirements.txt