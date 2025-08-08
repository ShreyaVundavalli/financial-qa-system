#!/bin/bash
echo " Setting up Financial Q&A System Environment"

# Create virtual environment
echo " Creating virtual environment..."
python3 -m venv venv

# Activate the environment
echo " Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆ Upgrading pip..."
pip install --upgrade pip

# Install required packages
echo "📚 Installing dependencies from requirements.txt..."
pip install -r requirements.txt

echo ""
echo "Virtual environment setup complete!"
echo ""
echo "📋 Next steps:"
echo "1. Activate environment: source venv/bin/activate"
echo "2. Update your email in src/data_acquisition.py (line 12)"
echo "3. Run data acquisition: python src/data_acquisition.py"
echo "4. Test the system: python src/main.py"
echo "5. Try interactive mode: python src/main.py --interactive"
