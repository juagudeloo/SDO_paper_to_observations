#!/bin/bash

# NASA ADS SDO API Setup Script
# This script sets up a virtual environment and installs the required dependencies

set -e  # Exit on any error

PROJECT_NAME="nasa-ads-sdo-api"
VENV_NAME="venv"
PYTHON_MIN_VERSION="3.8"

echo "🚀 Setting up NASA ADS SDO API..."

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not installed. Please install Python 3.${PYTHON_MIN_VERSION}+ and try again."
    exit 1
fi

# Check Python version
PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
if ! python3 -c "import sys; exit(0 if sys.version_info >= (3, 8) else 1)"; then
    echo "❌ Python 3.${PYTHON_MIN_VERSION}+ is required. Found version ${PYTHON_VERSION}"
    exit 1
fi

echo "✅ Python ${PYTHON_VERSION} found"

# Check if python3-venv is available (required on Ubuntu/Debian)
echo "🔍 Checking for python3-venv..."
if ! python3 -m venv --help &> /dev/null; then
    echo "❌ python3-venv is not installed"
    echo "📋 Please install it first:"
    echo "   sudo apt update"
    echo "   sudo apt install python3-venv"
    echo ""
    echo "After installation, run this setup script again."
    exit 1
fi

echo "✅ python3-venv is available"

# Create virtual environment
echo "📦 Creating virtual environment..."
if [ -d "$VENV_NAME" ]; then
    echo "⚠️  Virtual environment already exists. Removing it..."
    rm -rf "$VENV_NAME"
fi

python3 -m venv "$VENV_NAME"
echo "✅ Virtual environment created"

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source "$VENV_NAME/bin/activate"

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "📚 Installing dependencies..."
pip install -r requirements.txt

echo "✅ Dependencies installed successfully"

# Create executable script
echo "🔧 Creating executable script..."
cat > run_api.sh << 'EOF'
#!/bin/bash

# NASA ADS SDO API Runner Script
# This script activates the virtual environment and starts the API server

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found. Please run setup.sh first."
    exit 1
fi

# Activate virtual environment
source venv/bin/activate

# Change to API directory
cd api/scripts

# Check if database exists
if [ ! -f "../database/sdo_papers_2010_2024.db" ]; then
    echo "❌ Database file not found. Please ensure the database is in api/database/sdo_papers_2010_2024.db"
    exit 1
fi

# Start the API server
echo "🚀 Starting NASA ADS SDO API server..."
echo "📍 API will be available at: http://localhost:8000"
echo "📖 API documentation at: http://localhost:8000/docs"
echo "🛑 Press Ctrl+C to stop the server"
echo ""

uvicorn main:app --host 0.0.0.0 --port 8000 --reload
EOF

chmod +x run_api.sh

echo "✅ Executable script created"

# Create development script
echo "🔧 Creating development script..."
cat > run_dev.sh << 'EOF'
#!/bin/bash

# NASA ADS SDO API Development Runner Script
# This script runs the API in development mode with auto-reload

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found. Please run setup.sh first."
    exit 1
fi

# Activate virtual environment
source venv/bin/activate

# Change to API directory
cd api/scripts

# Start the API server in development mode
echo "🚀 Starting NASA ADS SDO API server in development mode..."
echo "📍 API will be available at: http://localhost:8000"
echo "📖 API documentation at: http://localhost:8000/docs"
echo "🔄 Auto-reload enabled for development"
echo "🛑 Press Ctrl+C to stop the server"
echo ""

uvicorn main:app --host 127.0.0.1 --port 8000 --reload --log-level debug
EOF

chmod +x run_dev.sh

echo ""
echo "🎉 Setup completed successfully!"
echo ""
echo "📋 Next steps:"
echo "   1. Run './run_api.sh' to start the API server"
echo "   2. Or run './run_dev.sh' for development mode"
echo "   3. Access the API at http://localhost:8000"
echo "   4. View API documentation at http://localhost:8000/docs"
echo ""
echo "🔧 Scripts created:"
echo "   - run_api.sh: Production server"
echo "   - run_dev.sh: Development server with auto-reload"
echo ""