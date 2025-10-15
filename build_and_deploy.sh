#!/bin/bash

# Build and deploy script for Sphinx documentation
echo "Building Sphinx documentation..."

# Navigate to docs directory
cd docs

# Build the documentation
make html

# Go back to root directory
cd ..

# Add all changes
git add .

# Commit changes
git commit -m "Update documentation"

# Push to main branch
git push origin main

echo "Documentation built and pushed to GitHub!"
echo "GitHub Actions will automatically deploy to GitHub Pages."
echo "Your site should be available at: https://jiegroup.github.io/GenAI/"
