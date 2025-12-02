#!/bin/bash
# Setup script for Fused LLM Chatbot dependencies (Python 3.13 compatible)

echo "🔧 Setting up Fused LLM Chatbot dependencies for Python 3.13..."
echo "=" * 60

# Check Python version
echo "📋 Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "   Python: $python_version"

# Check if Python 3.13
if [[ $python_version == 3.13* ]]; then
    echo ""
    echo "⚠️  Python 3.13 detected!"
    echo "   PyTorch 2.3.0+ may not be available for Python 3.13 yet."
    echo "   Trying to install latest compatible PyTorch version..."
    echo ""
    
    # Try installing latest PyTorch (may be 2.2.x or earlier)
    echo "📦 Installing PyTorch (latest compatible version)..."
    pip install torch --index-url https://download.pytorch.org/whl/cpu 2>&1 | tail -5
    
    # Check if it worked
    python3 -c "import torch; print(f'✅ PyTorch {torch.__version__} installed')" 2>/dev/null || {
        echo ""
        echo "❌ PyTorch installation failed for Python 3.13"
        echo ""
        echo "💡 Solutions:"
        echo "   1. Use Python 3.11 or 3.12 (recommended):"
        echo "      conda create -n fused_llm python=3.11"
        echo "      conda activate fused_llm"
        echo "      pip install torch>=2.3.0"
        echo ""
        echo "   2. Try installing from nightly builds:"
        echo "      pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cpu"
        echo ""
        echo "   3. Wait for official PyTorch 3.13 support"
        exit 1
    }
else
    # Normal installation for Python 3.11/3.12
    echo "📦 Installing PyTorch..."
    pip install torch>=2.3.0
fi

# Install chatbot dependencies
echo ""
echo "📦 Installing chatbot dependencies..."
pip install openai requests

# Install server dependencies (with relaxed torch requirement for Python 3.13)
echo ""
echo "📦 Installing server dependencies..."
if [ -d "vattention/sarathi-lean" ]; then
    cd vattention/sarathi-lean
    
    # Create a modified requirements file for Python 3.13
    if [[ $python_version == 3.13* ]]; then
        echo "   Creating Python 3.13 compatible requirements..."
        cp requirements.txt requirements.txt.backup
        sed 's/torch >= 2.3.0/torch >= 2.0.0/' requirements.txt > requirements_py313.txt
        pip install -r requirements_py313.txt || pip install -r requirements.txt.backup
        mv requirements.txt.backup requirements.txt
    else
        pip install -r requirements.txt
    fi
    
    echo "✅ Server dependencies installed"
    cd ../..
else
    echo "⚠️  vattention/sarathi-lean directory not found"
fi

echo ""
echo "✅ Setup complete!"
echo ""
echo "💡 Test your setup:"
echo "   python fused_chatbot.py --test-server"



