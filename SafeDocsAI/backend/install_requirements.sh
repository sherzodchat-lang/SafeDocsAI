#!/bin/bash
echo "Installing heavy dependencies (this may take a while per package)..."

echo "1. Installing torch (via pip)..."
pip install torch --default-timeout=1000

echo "2. Installing sentence-transformers..."
pip install sentence-transformers --default-timeout=1000

echo "Done! Restart backend if needed."
