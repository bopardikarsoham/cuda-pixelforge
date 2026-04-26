#!/bin/bash

echo "Building project..."
make build

echo "Running project..."
./batch_processor input output

echo "Done. Check output/ and artifacts/"