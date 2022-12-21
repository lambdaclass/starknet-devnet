#!/bin/bash

set -e

echo "npm: $(npm --version)"
echo "npm: $(node --version)"
echo "pip: $(pip --version)"
echo "pip3: $(pip3 --version)"
echo "python: $(python --version)"
echo "python3: $(python3 --version)"

pip3 install -U poetry==1.2.1
echo "poetry: $(poetry --version)"

# install dependencies
poetry install
poetry lock --check
npm ci

# install cairo-rs-py
git clone git@github.com:lambdaclass/cairo-rs-py.git
poetry run maturin develop --release -m cairo-rs-py/Cargo.toml --no-default-features --features extension
