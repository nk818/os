# Quick Fix for Python 3.13 Issue

You're using Python 3.13, but PyTorch 2.3.0+ isn't available for it yet.

## Quick Solution (2 minutes):

```bash
# Create Python 3.11 environment
conda create -n fused_llm python=3.11 -y
conda activate fused_llm

# Go to project directory
cd /Users/nk/Desktop/OS

# Install dependencies
./setup_dependencies.sh

# Test it
python fused_chatbot.py --test-server
```

That's it! The chatbot should work now.

For more details, see PYTHON_VERSION_GUIDE.md
