# How to Use the Standalone Chatbot

## 🚀 Quick Start - 3 Ways to Use

### Method 1: Interactive CLI (Easiest) ⭐

**Just run this command:**

```bash
cd /Users/nk/Desktop/OS/standalone_chatbot
conda activate fused_llm
python standalone_chat.py --model gpt2 --interactive
```

Then type your questions directly in the terminal!

**Example:**
```
You: What is machine learning?
Assistant: Machine learning is a method of data analysis...

You: Tell me about Python
Assistant: Python is a high-level programming language...
```

**To exit:** Type `quit` or `exit`

---

### Method 2: API Server (For Web/Apps)

**Step 1: Start the server**

```bash
cd /Users/nk/Desktop/OS/standalone_chatbot
conda activate fused_llm
python api_server.py --model gpt2 --port 8000
```

**Step 2: Ask questions via API**

**Using curl:**
```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "What is AI?"}],
    "model": "gpt2"
  }'
```

**Using Python:**
```python
import requests

response = requests.post(
    "http://localhost:8000/v1/chat/completions",
    json={
        "messages": [{"role": "user", "content": "What is machine learning?"}],
        "model": "gpt2"
    }
)

print(response.json()["choices"][0]["message"]["content"])
```

**API Endpoints:**
- Chat: `http://localhost:8000/v1/chat/completions`
- Health: `http://localhost:8000/health`
- Docs: `http://localhost:8000/docs` (Interactive API documentation)

---

### Method 3: Single Question

**Ask one question at a time:**

```bash
python standalone_chat.py --model gpt2 --prompt "What is artificial intelligence?"
```

---

## 📍 Where to Run Commands

**Open Terminal and navigate to:**
```bash
cd /Users/nk/Desktop/OS/standalone_chatbot
```

**Make sure you're in the right environment:**
```bash
conda activate fused_llm
```

---

## 🎯 Recommended: Interactive Mode

**Just run:**
```bash
python standalone_chat.py --model gpt2 --interactive
```

Then start chatting! Type your questions and press Enter.

---

## 🔧 With Optimizations

**With AdaptiVocab (tokenizer optimization):**
```bash
python standalone_chat.py \
    --model gpt2 \
    --patch-tokenizer path/to/patch_tokenizer.pkl \
    --interactive
```

**With LaRoSA (activation sparsity):**
```bash
python standalone_chat.py \
    --model gpt2 \
    --larosa-sparsity 0.4 \
    --interactive
```

---

## ❓ Troubleshooting

**If you get "command not found":**
- Make sure you're in the right directory: `cd /Users/nk/Desktop/OS/standalone_chatbot`
- Make sure conda environment is activated: `conda activate fused_llm`

**If the model is slow:**
- This is normal on CPU
- Use smaller models (gpt2 instead of gpt2-medium)
- Reduce max_new_tokens in the code if needed

**If you get import errors:**
```bash
pip install -r requirements.txt
```

---

## 📝 Example Session

```bash
$ python standalone_chat.py --model gpt2 --interactive

============================================================
🤖 Standalone Fused LLM Chatbot
============================================================
Type 'quit' or 'exit' to end the conversation

You: Hello!
Assistant: Hello! How can I help you today?

You: What is Python?
Assistant: Python is a high-level programming language known for its simplicity and readability.

You: Tell me a joke
Assistant: Why don't scientists trust atoms? Because they make up everything!

You: quit
👋 Goodbye!
```

---

**That's it! Just run the interactive command and start chatting!** 🎉

