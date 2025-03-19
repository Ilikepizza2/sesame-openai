#!/bin/bash

START_COMMAND="python api-server.py"


# Create and activate Conda environment
echo "Setting up Conda environment..."
conda create -y -n "sesame" python="3.10"
source activate "sesame" || conda activate "sesame"

# Install dependencies
echo "Installing dependencies..."
pip install git+https://github.com/senstella/csm-mlx
pip install -r requirements.txt


# Start the model server
echo "Starting Zonos server..."
eval "$START_COMMAND"