#!/bin/bash

# Color codes for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Starting setup...${NC}"

# Step 1: Setting up Python virtual environment
echo -e "${YELLOW}Setting up Python virtual environment...${NC}"
python3 -m venv venv
source venv/bin/activate

# Step 2: Installing PyTorch based on CUDA availability
echo -e "${YELLOW}Detecting CUDA availability...${NC}"
if command -v nvidia-smi &> /dev/null; then
    echo -e "${GREEN}CUDA is available. Installing PyTorch with CUDA support...${NC}"
    pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
else
    echo -e "${RED}CUDA not found. Installing CPU-only version of PyTorch...${NC}"
    pip install torch torchvision torchaudio
fi

# Step 3: Installing dependencies from requirements.txt
echo -e "${YELLOW}Installing dependencies from requirements.txt...${NC}"
if [[ -f "requirements.txt" ]]; then
    pip install -r requirements.txt
else
    echo -e "${RED}Error: requirements.txt not found!${NC}"
    exit 1
fi

# Step 4: Validating all dependencies
echo -e "${YELLOW}Validating installed dependencies...${NC}"
pip check

# Step 5: Creating project directory structure
echo -e "${YELLOW}Creating project directory structure...${NC}"
mkdir -p data/models data/logs

# Step 6: Displaying hardware information
echo -e "${YELLOW}Hardware Information:${NC}"
lscpu
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi
fi

# Final setup instructions
echo -e "${GREEN}Setup complete! Please activate the virtual environment with 'source venv/bin/activate'.${NC}"
echo -e "${GREEN}For Jupyter Notebook users, don't forget to install the kernel:${NC}"
echo -e "${GREEN}python -m ipykernel install --user --name=venv${NC}"
