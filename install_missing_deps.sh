#!/bin/bash
# Quick script to install missing dependencies

echo "📦 Installing missing dependencies..."

# Install critical dependencies
pip install numpy transformers fastapi uvicorn

# Install other server dependencies
cd vattention/sarathi-lean
pip install packaging ninja psutil ray pandas pyarrow sentencepiece matplotlib plotly_express seaborn wandb kaleido jupyterlab pillow tiktoken grpcio openai 2>&1 | grep -E "(Requirement|Successfully|ERROR)" | head -20

echo ""
echo "✅ Dependencies installed!"
echo ""
echo "💡 Test: python fused_chatbot.py --test-server"
