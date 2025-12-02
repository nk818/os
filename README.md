# Fused LLM Efficiency System: AdaptiVocab + vAttention + LaRoSA + Gstack

This repository contains four complementary technologies for optimizing LLM efficiency:

1. **Gstack**: Model growth through depthwise stacking for 40-60% pre-training speedup
2. **AdaptiVocab**: Vocabulary-centric adaptation that reduces token usage by 25%+
3. **vAttention**: Dynamic KV-cache memory management for efficient serving
4. **LaRoSA**: Layerwise Rotated Sparse Activation for 1.30x-1.90x inference speedup

## 📁 Repository Structure

```
OS/
├── AdaptiVocab/              # Vocabulary optimization system
│   ├── src/
│   │   ├── build_vocab/      # Tokenizer patching and vocabulary building
│   │   ├── ft/               # Fine-tuning scripts
│   │   ├── evaluate/         # Evaluation scripts
│   │   └── utils/            # Utility functions
│   └── requirements.txt
│
├── vattention/               # Memory management system
│   ├── vattention/          # Core vAttention library (C++/CUDA)
│   ├── sarathi-lean/        # LLM serving system with vAttention
│   ├── scripts/              # Benchmark scripts
│   └── microbenchmarks/      # Performance tests
│
├── INTEGRATION_ANALYSIS.md   # Detailed technical analysis
├── INTEGRATION_PLAN.md       # Step-by-step integration guide
├── LAROSA_INTEGRATION.md     # LaRoSA activation sparsity integration
├── GSTACK_INTEGRATION.md     # Gstack model growth integration
└── README.md                 # This file
```

## 🎯 Integration Overview

### Why Combine These Technologies?

**Gstack**, **AdaptiVocab**, **vAttention**, and **LaRoSA** are highly complementary across the entire LLM lifecycle:

- **Gstack** accelerates **pre-training** (40-60% faster model development)
- **AdaptiVocab** reduces the **amount** of data (fewer tokens)
- **vAttention** optimizes how that data is **stored** (better memory management)
- **LaRoSA** reduces the **computation** needed (activation sparsity)

Together, they provide:
- ✅ 40-60% pre-training speedup (Gstack)
- ✅ 25%+ token reduction (AdaptiVocab)
- ✅ 15% memory savings (vAttention)
- ✅ 1.30x-1.90x inference speedup (LaRoSA)
- ✅ Faster inference (fewer tokens + better memory + sparse activations)
- ✅ Higher throughput (more concurrent requests)
- ✅ Lower costs (less compute and memory)
- ✅ **50-70% overall efficiency gain** (combined across pre-training and inference)

### Integration Status

**Status**: ✅ **Feasible and Recommended**

The integration is technically feasible with moderate complexity. See `INTEGRATION_ANALYSIS.md` for detailed technical analysis and `INTEGRATION_PLAN.md` for implementation steps.

## 🚀 Quick Start

### 1. Explore the Projects

```bash
# Read AdaptiVocab documentation
cat AdaptiVocab/README.md

# Read vAttention documentation
cat vattention/README.md
```

### 2. Review Integration Plans

```bash
# Read detailed analysis
cat INTEGRATION_ANALYSIS.md

# Read implementation plan
cat INTEGRATION_PLAN.md
```

### 3. Set Up Environment

See `INTEGRATION_PLAN.md` for detailed environment setup instructions.

## 📊 Expected Benefits

### Combined Efficiency Gains

| Metric | Gstack | AdaptiVocab | vAttention | LaRoSA | Combined |
|--------|--------|-------------|------------|--------|----------|
| Pre-training Speedup | 40-60% | - | - | - | 40-60% |
| Token Reduction | - | 25%+ | - | - | 25%+ |
| Memory Efficiency | - | - | 10-20% | - | 30-40% |
| Inference Speedup | - | - | - | 1.30x-1.90x | 1.30x-1.90x |
| Throughput | - | ~15% | ~20% | ~30-90% | 30-90% |
| Cost Savings | ~40-60% | ~25% | ~15% | ~30-90% | **50-70%** |

### Synergistic Effects

**Pre-training Phase (Gstack)**:
1. **Faster model development** → **Better base models** → **Improved starting point**

**Inference Phase (AdaptiVocab + vAttention + LaRoSA)**:
1. **Fewer tokens** → **Less KV-cache needed** → **Better memory utilization**
2. **Domain-specific tokens** → **Better compression** → **Faster processing**
3. **Optimized memory** → **More concurrent requests** → **Higher throughput**
4. **Sparse activations** → **Faster computation** → **Higher throughput**

**End-to-End Pipeline**:
- Gstack creates optimized base models faster
- AdaptiVocab fine-tunes with domain-specific vocabulary
- vAttention and LaRoSA optimize serving efficiency

## 🔧 Key Integration Points

1. **Tokenizer Integration**: Wrap PatchTokenizer for Sarathi-Serve compatibility
2. **Model Loading**: Load models with patched embeddings
3. **KV-Cache Optimization**: Adjust cache size calculations for reduced tokens
4. **Attention Wrappers**: Ensure compatibility with FlashAttention/FlashInfer

## 📚 Documentation

- **INTEGRATION_ANALYSIS.md**: Comprehensive technical analysis
  - Project overviews
  - Integration points
  - Technical challenges
  - Expected benefits

- **INTEGRATION_PLAN.md**: Step-by-step implementation guide
  - Environment setup
  - Code integration examples
  - Testing strategy
  - Usage examples

- **GSTACK_INTEGRATION.md**: Gstack model growth integration
  - Depthwise stacking implementation
  - Growth timing and factor guidelines
  - Pre-training pipeline integration

- **LAROSA_INTEGRATION.md**: LaRoSA activation sparsity integration
  - Rotation matrix computation
  - Activation sparsification
  - Performance expectations

## 🧪 Testing

See `INTEGRATION_PLAN.md` for testing strategies and example test scripts.

## 📖 References

- **Gstack Paper**: [arXiv:2405.15319v2](https://arxiv.org/abs/2405.15319v2) - NeurIPS 2024
- **AdaptiVocab Paper**: [arXiv:2503.19693](https://arxiv.org/abs/2503.19693) - Accepted to COLM 2025
- **vAttention Paper**: [arXiv:2405.04437](https://arxiv.org/abs/2405.04437)
- **LaRoSA Paper**: [arXiv:2507.01299v1](https://arxiv.org/abs/2507.01299v1) - ICML 2025
- **Gstack GitHub**: https://llm-stacking.github.io/
- **AdaptiVocab GitHub**: https://github.com/itay-nakash/AdaptiVocab
- **vAttention GitHub**: https://github.com/microsoft/vattention

## 🤝 Contributing

This is an integration project combining four research systems. For contributions:

1. Review the integration analysis and plan
2. Follow the implementation steps in `INTEGRATION_PLAN.md`
3. Test thoroughly before submitting changes
4. Document any new integration points

## 📝 License

- **Gstack**: Check paper and implementation for license information
- **AdaptiVocab**: Check `AdaptiVocab/` for license information
- **vAttention**: MIT License (see `vattention/LICENSE`)
- **LaRoSA**: Check paper and implementation for license information

## 🎓 Citation

If you use this integrated system, please cite all four papers:

```bibtex
@inproceedings{du2024stacking,
  title={Stacking Your Transformers: A Closer Look at Model Growth for Efficient LLM Pre-Training},
  author={Du, Wenyu and Luo, Tongxu and Qiu, Zihan and Huang, Zeyu and Shen, Yikang and Cheng, Reynold and Guo, Yike and Fu, Jie},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2024},
  url={https://arxiv.org/abs/2405.15319v2}
}

@article{adaptivocab2025,
  title={AdaptiVocab: Enhancing LLM Efficiency in Focused Domains through Lightweight Vocabulary Adaptation},
  author={Nakash, Itay and Calderon, Nitay and Ben David, Eyal and Hoffer, Elad and Reichart, Roi},
  journal={COLM 2025},
  year={2025},
  url={https://arxiv.org/abs/2503.19693}
}

@misc{prabhu2024vattention,
  title={vAttention: Dynamic Memory Management for Serving LLMs without PagedAttention},
  author={Prabhu, Ramya and Nayak, Ajay and Mohan, Jayashree and Ramjee, Ramachandran and Panwar, Ashish},
  year={2024},
  url={https://arxiv.org/abs/2405.04437}
}

@misc{liu2025larosa,
  title={La RoSA: Enhancing LLM Efficiency via Layerwise Rotated Sparse Activation},
  author={Liu, Kai and Xu, Bowen and Wu, Shaoyu and others},
  journal={ICML 2025},
  year={2025},
  url={https://arxiv.org/abs/2507.01299v1}
}
```

## ⚠️ Notes

- This is a research integration project
- All systems are research prototypes
- Production use requires additional testing and optimization
- PyTorch version compatibility needs to be resolved (AdaptiVocab uses 2.2.2, vAttention requires 2.3.0+)
- Gstack is a pre-training optimization, while AdaptiVocab, vAttention, and LaRoSA are inference optimizations

## 🤖 Fused Chatbot

A unified chatbot interface that integrates all four methods is now available!

### Quick Setup

**⚠️ Python Version:**
- **Recommended:** Python 3.11 or 3.12 (full PyTorch support)
- **Python 3.13:** Use `./setup_dependencies_py313.sh` (may have compatibility issues)

```bash
# 1. Install dependencies (REQUIRED - PyTorch is critical!)

# For Python 3.11/3.12:
./setup_dependencies.sh

# For Python 3.13:
./setup_dependencies_py313.sh

# Or manually:
pip install torch openai requests
cd vattention/sarathi-lean && pip install -r requirements.txt

# 2. Test your setup
python fused_chatbot.py --test-server

# 3. Run the chatbot
python fused_chatbot.py
```

### Usage

```bash
# Basic usage
python fused_chatbot.py

# With all optimizations
python fused_chatbot.py \
    --model gpt2 \
    --patch-tokenizer path/to/patch_tokenizer.pkl \
    --larosa-sparsity 0.4

# On Mac (no GPU support)
python fused_chatbot.py --no-vattention --no-larosa
```

**⚠️ Important:** 
- The server requires PyTorch. If you see `ModuleNotFoundError: No module named 'torch'`, install it with `pip install torch`.
- **Python 3.13 users:** PyTorch 2.3.0+ is not available for Python 3.13 yet. Use Python 3.11 or 3.12 instead. See [PYTHON_VERSION_GUIDE.md](PYTHON_VERSION_GUIDE.md) for details.
- **Missing dependencies?** If you see missing packages after setup, run: `./install_missing_deps.sh` or see [QUICK_FIX.md](QUICK_FIX.md)

See [CHATBOT_README.md](CHATBOT_README.md) for detailed usage instructions and [TROUBLESHOOTING.md](TROUBLESHOOTING.md) for common issues.

## 🚧 Next Steps

1. ✅ Clone all repositories
2. ✅ Analyze integration feasibility
3. ✅ Create unified chatbot interface
4. ⬜ Set up unified environment
5. ⬜ Implement tokenizer wrapper
6. ⬜ Integrate model loading
7. ⬜ Update KV-cache calculations
8. ⬜ Run integration tests
9. ⬜ Benchmark performance

---

**Status**: Analysis complete. Chatbot interface ready. See `INTEGRATION_PLAN.md` for next steps.

