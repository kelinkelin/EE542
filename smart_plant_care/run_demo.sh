#!/bin/bash

# Ensure we are in the script's directory or handle paths correctly
cd "$(dirname "$0")"

echo "🌿 Generating Sci-Fi Demo Interface..."
./venv/bin/python generate_cool_demo.py

echo "🚀 Launching Demo..."
if [[ "$OSTYPE" == "darwin"* ]]; then
    open docs/cool_demo.html
elif [[ "$OSTYPE" == "cygwin" ]]; then
    cygstart docs/cool_demo.html
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    xdg-open docs/cool_demo.html
else
    # Windows mostly
    start docs/cool_demo.html
fi

