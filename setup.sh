#!/bin/bash

# Function to check for system requirements
check_requirements() {
    echo "Checking system requirements..."
    # Check for Python
    if ! command -v python3 &>/dev/null; then
        echo "Python 3 is not installed. Please install Python 3 and try again."
        exit 1
    fi

    # Check for pip
    if ! command -v pip &>/dev/null; then
        echo "pip is not installed. Please install pip and try again."
        exit 1
    fi
}

# Function to setup Python virtual environment
setup_virtualenv() {
    echo "Setting up Python virtual environment..."
    python3 -m venv venv
    source venv/bin/activate
}

# Function to install requirements
install_requirements() {
    echo "Installing requirements..."
    pip install -r requirements.txt
}

# Function to validate installation
validate_installation() {
    echo "Validating installation..."
    if pip show torch &>/dev/null; then
        echo "Torch is successfully installed!"
    else
        echo "Failed to install Torch. Please check the error messages above and try again."
        exit 1
    fi

    if pip show pipreqs &>/dev/null; then
        echo "pipreqs is successfully installed!"
    else
        echo "Failed to install pipreqs. Please check the error messages above and try again."
        exit 1
    fi
}

# Main script execution
check_requirements
setup_virtualenv
install_requirements
validate_installation

# CUDA detection
if [[ $(python3 -c 'import torch; print(torch.cuda.is_available())') == "True" ]]; then
    echo "CUDA is available!"
else
    echo "CUDA is not available. Falling back to CPU."
fi
