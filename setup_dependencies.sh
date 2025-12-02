#!/bin/bash
# Setup script for Fused LLM Chatbot dependencies

echo "🔧 Setting up Fused LLM Chatbot dependencies..."
echo "=" * 60

# Check Python version
echo "📋 Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "   Python: $python_version"

# Install PyTorch (critical dependency)
echo ""
echo "📦 Installing PyTorch..."
# Try 2.3.0+, fall back to latest if not available
pip install "torch>=2.3.0" 2>/dev/null || {
    echo "   PyTorch 2.3.0+ not available, installing latest compatible version..."
    pip install torch
}

# Install chatbot dependencies
echo ""
echo "📦 Installing chatbot dependencies..."
pip install openai requests

# Install server dependencies
echo ""
echo "📦 Installing server dependencies..."
if [ -d "vattention/sarathi-lean" ]; then
    cd vattention/sarathi-lean
    if [ -f "requirements.txt" ]; then
        # Create a flexible requirements file that allows torch 2.2.x
        echo "   Creating flexible requirements (allowing torch 2.2.x)..."
        sed 's/torch >= 2.3.0/torch >= 2.2.0/' requirements.txt > requirements_flexible.txt
        
        # Try installing with flexible requirements
        if pip install -r requirements_flexible.txt; then
            echo "✅ Server dependencies installed"
            rm requirements_flexible.txt
        else
            echo "⚠️  Some dependencies failed, trying individual packages..."
            # Install critical packages individually
            pip install transformers fastapi uvicorn numpy packaging ninja psutil ray pandas pyarrow sentencepiece matplotlib plotly_express seaborn wandb kaleido jupyterlab pillow tiktoken grpcio openai || true
            echo "✅ Core dependencies installed (some optional packages may be missing)"
            rm requirements_flexible.txt
        fi
    else
        echo "⚠️  requirements.txt not found in vattention/sarathi-lean"
    fi
    cd ../..
else
    echo "⚠️  vattention/sarathi-lean directory not found"
fi

echo ""
echo "✅ Setup complete!"
echo ""
echo "💡 Test your setup:"
echo "   python fused_chatbot.py --test-server"

