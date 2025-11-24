#!/bin/bash

# Script to set up Python virtual environment and install dependencies

echo "Setting up Python 3.10 virtual environment..."

# Check if Python 3.10 is available
if ! command -v python3.10 &> /dev/null
then
    echo "Error: Python 3.10 is not installed or not in PATH"
    echo "Please install Python 3.10 first"
    exit 1
fi

# Create virtual environment with Python 3.10
python3.10 -m venv img_classifier_env

# Check if venv was created successfully
if [ ! -d "venv" ]; then
    echo "Error: Failed to create virtual environment"
    exit 1
fi

echo "Virtual environment created successfully!"

# Activate virtual environment
source img_classifier_env/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements
if [ -f "requirements.txt" ]; then
    echo "Installing requirements from requirements.txt..."
    pip install -r requirements.txt
    echo "All dependencies installed successfully!"
else
    echo "Warning: requirements.txt not found"
fi

echo ""
echo "Setup complete!"
echo "To activate the virtual environment, run: source img_classifier_env/bin/activate"
echo "To deactivate, run: deactivate"
