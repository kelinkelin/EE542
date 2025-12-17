#!/bin/bash

# Ensure we are in the script's directory
cd "$(dirname "$0")"

echo "🌿 Initializing LeafMind OS..."

# Check if venv exists
if [ ! -d "venv" ]; then
    echo "⚠️  Virtual environment not found!"
    exit 1
fi

# Install Flask if missing (silent check)
if ! ./venv/bin/pip show flask > /dev/null; then
    echo "📦 Installing System Dependencies (Flask)..."
    ./venv/bin/pip install flask pandas
fi

echo "🚀 Launching Neural Core..."
echo "👉 Open http://127.0.0.1:5000 in your browser"

# Run the app
./venv/bin/python web_app/app.py

