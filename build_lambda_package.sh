#!/bin/bash

# Exit on error
set -e

# Clean up any existing virtual environment
rm -rf lambda_venv deployment

# Create and activate a virtual environment
python3 -m venv lambda_venv
source lambda_venv/bin/activate

# Upgrade pip and install build tools
pip install --upgrade pip
pip install wheel setuptools

# Install dependencies
pip install -r lambda_requirements.txt

# Create deployment directory
mkdir -p deployment
cp lambda_handler.py deployment/

# Find the site-packages directory
SITE_PACKAGES_DIR=$(find lambda_venv/lib -name site-packages)

# Create the package directory
cd "$SITE_PACKAGES_DIR"
zip -r ../../../../deployment/lambda_package.zip .

# Add our lambda handler
cd ../../../../deployment
zip -g lambda_package.zip lambda_handler.py

# Clean up
cd ..
rm -rf lambda_venv deployment/lambda_handler.py

echo "Deployment package created at deployment/lambda_package.zip" 
