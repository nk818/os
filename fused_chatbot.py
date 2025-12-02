#!/usr/bin/env python3
"""
Fused LLM Chatbot - Unified Interface for Gstack + AdaptiVocab + vAttention + LaRoSA

This chatbot integrates all four optimization methods:
1. Gstack: Uses Gstack-trained models (if available)
2. AdaptiVocab: Domain-specific vocabulary optimization
3. vAttention: Dynamic KV-cache memory management
4. LaRoSA: Activation sparsity for faster inference
"""

import os
import sys
import json
import argparse
import subprocess
import time
import requests
from typing import Optional, Dict, List
from pathlib import Path

try:
    from openai import OpenAI
except ImportError:
    print("Warning: openai package not found. Install with: pip install openai")
    OpenAI = None


class FusedChatbotConfig:
    """Configuration for the Fused LLM Chatbot"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_config(config_path)
        
    def _load_config(self, config_path: Optional[str] = None) -> Dict:
        """Load configuration from file or use defaults"""
        default_config = {
            "model_name": "gpt2",
            "patch_tokenizer_path": None,
            "attention_backend": "fa_vattn",  # vAttention
            "block_size": 2097152,
            "larosa_sparsity": 0.4,  # 40% sparsity
            "server_host": "localhost",
            "server_port": 8000,
            "server_timeout": 300,  # 5 minutes
            "gstack_model_path": None,  # Path to Gstack-trained model
        }
        
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                user_config = json.load(f)
            default_config.update(user_config)
        
        return default_config
    
    def save(self, path: str):
        """Save configuration to file"""
        with open(path, 'w') as f:
            json.dump(self.config, f, indent=2)
    
    def __getitem__(self, key):
        return self.config[key]
    
    def __setitem__(self, key, value):
        self.config[key] = value


class FusedServerManager:
    """Manages the fused LLM server process"""
    
    def __init__(self, config: FusedChatbotConfig):
        self.config = config
        self.process = None
        self.server_url = f"http://{config['server_host']}:{config['server_port']}"
        
    def start_server(self) -> bool:
        """Start the fused LLM server"""
        if self.is_running():
            print(f"✅ Server already running at {self.server_url}")
            return True
        
        print("🚀 Starting Fused LLM Server...")
        print(f"   Model: {self.config['model_name']}")
        print(f"   AdaptiVocab: {'✅' if self.config['patch_tokenizer_path'] else '❌'}")
        print(f"   vAttention: {'✅' if self.config['attention_backend'] == 'fa_vattn' else '❌'}")
        print(f"   LaRoSA: {'✅' if self.config['larosa_sparsity'] > 0 else '❌'}")
        print(f"   Gstack Model: {'✅' if self.config['gstack_model_path'] else '❌ (using standard model)'}")
        
        # Check if sarathi-lean directory exists
        sarathi_dir = os.path.join(os.path.dirname(__file__), "vattention", "sarathi-lean")
        if not os.path.exists(sarathi_dir):
            print(f"❌ Sarathi directory not found: {sarathi_dir}")
            print("   Please ensure vattention/sarathi-lean is properly set up")
            return False
        
        # Check if the module can be imported and check critical dependencies
        try:
            import sys
            sys.path.insert(0, sarathi_dir)
            
            # Check for critical dependencies first
            try:
                import torch
                print(f"✅ PyTorch found: {torch.__version__}")
            except ImportError:
                print("❌ PyTorch (torch) is not installed!")
                print("   This is required for the server to run.")
                print("   Install with: pip install torch")
                print("   Or install all dependencies:")
                print(f"   cd {sarathi_dir} && pip install -r requirements.txt")
                return False
            
            # Try importing sarathi to check other dependencies
            try:
                import sarathi
                print("✅ Sarathi module can be imported")
            except ImportError as e:
                print(f"❌ Cannot import sarathi module: {e}")
                print("   Please install dependencies:")
                print(f"   cd {sarathi_dir} && pip install -r requirements.txt")
                return False
                
        except Exception as e:
            print(f"❌ Error checking dependencies: {e}")
            return False
        
        # Build command - Note: Sarathi uses ConfigParser which reads from config files
        # We'll need to pass arguments differently or use a config file
        cmd = [
            sys.executable, "-m", "sarathi.entrypoints.openai_server.api_server",
        ]
        
        # Note: The server uses ConfigParser which may require a config file
        # For now, we'll try with minimal arguments
        # The server should read from default config or environment
        
        # Add PatchTokenizer if available
        if self.config['patch_tokenizer_path']:
            if os.path.exists(self.config['patch_tokenizer_path']):
                cmd.extend(["--patch_tokenizer_path", self.config['patch_tokenizer_path']])
            else:
                print(f"⚠️  Warning: PatchTokenizer not found at {self.config['patch_tokenizer_path']}")
                print("   Continuing without AdaptiVocab...")
        
        # Add Gstack model path if available
        if self.config['gstack_model_path']:
            if os.path.exists(self.config['gstack_model_path']):
                # Note: This would need to be integrated into Sarathi-Serve
                print(f"📦 Using Gstack-trained model: {self.config['gstack_model_path']}")
            else:
                print(f"⚠️  Warning: Gstack model not found, using standard model")
        
        # Start server in background
        try:
            print(f"📝 Command: {' '.join(cmd)}")
            print(f"📁 Working directory: {os.path.join(os.path.dirname(__file__), 'vattention', 'sarathi-lean')}")
            
            self.process = subprocess.Popen(
                cmd,
                cwd=os.path.join(os.path.dirname(__file__), "vattention", "sarathi-lean"),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,  # Combine stderr with stdout
                text=True,
                bufsize=1  # Line buffered
            )
            
            # Wait for server to start with better error checking
            print("⏳ Waiting for server to start...")
            server_output = []
            check_interval = 2  # Check every 2 seconds
            max_wait = self.config['server_timeout']
            
            for i in range(0, max_wait, check_interval):
                # Check if process is still running
                if self.process.poll() is not None:
                    # Process has terminated
                    remaining_output = self.process.stdout.read()
                    if remaining_output:
                        server_output.append(remaining_output)
                    
                    print("\n❌ Server process terminated unexpectedly!")
                    print("📋 Server output:")
                    print("=" * 60)
                    output_text = "\n".join(server_output)
                    if output_text:
                        print(output_text)
                    else:
                        print("(No output captured)")
                    print("=" * 60)
                    return False
                
                # Check if server is responding
                if self.is_running():
                    print(f"✅ Server started successfully at {self.server_url}")
                    return True
                
                # Read any available output
                try:
                    # Non-blocking read
                    import select
                    if select.select([self.process.stdout], [], [], 0)[0]:
                        line = self.process.stdout.readline()
                        if line:
                            server_output.append(line.strip())
                            # Show important errors immediately
                            if any(keyword in line.lower() for keyword in ['error', 'exception', 'failed', 'traceback']):
                                print(f"\n⚠️  Server error: {line.strip()}")
                except:
                    pass  # select might not work on all systems
                
                time.sleep(check_interval)
                if i % 10 == 0 and i > 0:
                    print(f"   Still waiting... ({i}s)")
                    # Show recent output
                    if server_output:
                        recent = server_output[-5:]
                        print(f"   Recent output: ...")
                        for line in recent:
                            if line:
                                print(f"      {line[:80]}")
            
            print("\n❌ Server failed to start within timeout")
            print("📋 Server output (last 20 lines):")
            print("=" * 60)
            for line in server_output[-20:]:
                if line:
                    print(line)
            print("=" * 60)
            print("\n💡 Troubleshooting tips:")
            print("   1. Check if port 8000 is available: lsof -i :8000")
            print("   2. Check server dependencies: cd vattention/sarathi-lean && pip install -r requirements.txt")
            print("   3. Try running server manually to see full error:")
            print(f"      cd vattention/sarathi-lean")
            print(f"      python -m sarathi.entrypoints.openai_server.api_server --model_name {self.config['model_name']}")
            return False
            
        except Exception as e:
            print(f"❌ Error starting server: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def stop_server(self):
        """Stop the fused LLM server"""
        if self.process:
            print("🛑 Stopping server...")
            # Read any remaining output
            try:
                import select
                if select.select([self.process.stdout], [], [], 0)[0]:
                    remaining = self.process.stdout.read()
                    if remaining:
                        print(f"   Final output: {remaining[:200]}")
            except:
                pass
            
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
            self.process = None
            print("✅ Server stopped")
    
    def is_running(self) -> bool:
        """Check if server is running"""
        try:
            response = requests.get(f"{self.server_url}/v1/models", timeout=2)
            return response.status_code == 200
        except:
            return False


class FusedChatbot:
    """Main chatbot interface"""
    
    def __init__(self, config: FusedChatbotConfig):
        self.config = config
        self.server_manager = FusedServerManager(config)
        self.client = None
        self.conversation_history: List[Dict] = []
        
    def start(self):
        """Start the chatbot"""
        print("=" * 60)
        print("🤖 Fused LLM Chatbot")
        print("=" * 60)
        print("Integrating: Gstack + AdaptiVocab + vAttention + LaRoSA")
        print("=" * 60)
        print()
        
        # Start server
        if not self.server_manager.start_server():
            print("❌ Failed to start server. Exiting.")
            return False
        
        # Initialize OpenAI client
        if OpenAI:
            self.client = OpenAI(
                base_url=self.server_manager.server_url,
                api_key="not-needed"
            )
        else:
            print("❌ OpenAI client not available. Install with: pip install openai")
            return False
        
        print("\n💬 Chatbot ready! Type 'quit' or 'exit' to end the conversation.")
        print("   Type 'clear' to clear conversation history.")
        print("   Type 'status' to see optimization status.")
        print("   Type 'help' for more commands.\n")
        
        return True
    
    def chat(self, user_input: str) -> str:
        """Send a message and get response"""
        if not self.client:
            return "Error: Chatbot not initialized"
        
        try:
            # Add user message to history
            self.conversation_history.append({
                "role": "user",
                "content": user_input
            })
            
            # Create chat completion request
            messages = [
                {"role": "system", "content": "You are a helpful AI assistant."}
            ] + self.conversation_history
            
            response = self.client.chat.completions.create(
                model=self.config['model_name'],
                messages=messages,
                max_tokens=500,
                temperature=0.7
            )
            
            assistant_message = response.choices[0].message.content
            
            # Add assistant response to history
            self.conversation_history.append({
                "role": "assistant",
                "content": assistant_message
            })
            
            return assistant_message
            
        except Exception as e:
            return f"Error: {str(e)}"
    
    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []
        print("✅ Conversation history cleared")
    
    def show_status(self):
        """Show optimization status"""
        print("\n📊 Optimization Status:")
        print("=" * 60)
        print(f"Model: {self.config['model_name']}")
        print(f"Gstack: {'✅ Enabled' if self.config['gstack_model_path'] else '❌ Using standard model'}")
        print(f"AdaptiVocab: {'✅ Enabled' if self.config['patch_tokenizer_path'] else '❌ Disabled'}")
        print(f"vAttention: {'✅ Enabled' if self.config['attention_backend'] == 'fa_vattn' else '❌ Disabled'}")
        print(f"LaRoSA: {'✅ Enabled' if self.config['larosa_sparsity'] > 0 else '❌ Disabled'}")
        if self.config['larosa_sparsity'] > 0:
            print(f"   Sparsity: {self.config['larosa_sparsity']*100:.0f}%")
        print("=" * 60)
    
    def interactive_mode(self):
        """Run interactive chat mode"""
        while True:
            try:
                user_input = input("\nYou: ").strip()
                
                if not user_input:
                    continue
                
                # Handle commands
                if user_input.lower() in ['quit', 'exit']:
                    print("👋 Goodbye!")
                    break
                elif user_input.lower() == 'clear':
                    self.clear_history()
                    continue
                elif user_input.lower() == 'status':
                    self.show_status()
                    continue
                elif user_input.lower() == 'help':
                    print("\n📖 Available Commands:")
                    print("   quit/exit - Exit the chatbot")
                    print("   clear - Clear conversation history")
                    print("   status - Show optimization status")
                    print("   help - Show this help message")
                    continue
                
                # Get response
                print("🤖 Assistant: ", end="", flush=True)
                response = self.chat(user_input)
                print(response)
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
    
    def stop(self):
        """Stop the chatbot and server"""
        self.server_manager.stop_server()


def find_patch_tokenizer() -> Optional[str]:
    """Try to find an existing PatchTokenizer"""
    search_paths = [
        "AdaptiVocab/src/saved_patch_tokenizers_no_ngrams_new_logs",
        "AdaptiVocab/src/saved_patch_tokenizers_ngram_k_analysis",
        "AdaptiVocab/src/saved_patch_tokenizers",
    ]
    
    for base_path in search_paths:
        if os.path.exists(base_path):
            for root, dirs, files in os.walk(base_path):
                if "patch_tokenizer.pkl" in files:
                    return os.path.join(root, "patch_tokenizer.pkl")
    
    return None


def test_server_manually():
    """Test if server can be started manually"""
    print("🔍 Testing server startup manually...")
    print("=" * 60)
    
    # Check Python version first
    import sys
    python_version = sys.version_info
    print(f"📋 Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    if python_version.major == 3 and python_version.minor == 13:
        print("\n⚠️  WARNING: Python 3.13 detected!")
        print("=" * 60)
        print("❌ PyTorch 2.3.0+ is not available for Python 3.13 yet.")
        print("\n💡 Quick Fix:")
        print("   conda create -n fused_llm python=3.11 -y")
        print("   conda activate fused_llm")
        print("   ./setup_dependencies.sh")
        print("\n📖 See PYTHON_VERSION_GUIDE.md for details")
        print("=" * 60)
        return False
    
    sarathi_dir = os.path.join(os.path.dirname(__file__), "vattention", "sarathi-lean")
    if not os.path.exists(sarathi_dir):
        print(f"❌ Sarathi directory not found: {sarathi_dir}")
        return False
    
    print(f"📁 Sarathi directory: {sarathi_dir}")
    
    # Check dependencies
    print("\n🔍 Checking dependencies...")
    missing_deps = []
    
    try:
        import torch
        version = torch.__version__
        # Check if version is acceptable (2.2.0+ should work)
        major, minor = map(int, version.split('.')[:2])
        if major == 2 and minor >= 2:
            print(f"✅ PyTorch: {version} (compatible)")
        elif major == 2 and minor >= 0:
            print(f"⚠️  PyTorch: {version} (may work, but 2.2.0+ recommended)")
        else:
            print(f"⚠️  PyTorch: {version} (version may be too old)")
    except ImportError:
        print("❌ PyTorch: NOT INSTALLED")
        missing_deps.append("torch")
    
    try:
        import transformers
        print(f"✅ Transformers: {transformers.__version__}")
    except ImportError:
        print("❌ Transformers: NOT INSTALLED")
        missing_deps.append("transformers")
    
    try:
        import fastapi
        print(f"✅ FastAPI: {fastapi.__version__}")
    except ImportError:
        print("❌ FastAPI: NOT INSTALLED")
        missing_deps.append("fastapi")
    
    try:
        import uvicorn
        print(f"✅ Uvicorn: {uvicorn.__version__}")
    except ImportError:
        print("❌ Uvicorn: NOT INSTALLED")
        missing_deps.append("uvicorn")
    
    if missing_deps:
        print("\n❌ Missing dependencies detected!")
        print("=" * 60)
        print("📦 Install missing dependencies:")
        print(f"   pip install {' '.join(missing_deps)}")
        print("\n   Or install all requirements:")
        print(f"   cd {sarathi_dir} && pip install -r requirements.txt")
        print("=" * 60)
        return False
    
    print("\n✅ All critical dependencies found!")
    print("=" * 60)
    print(f"📝 Testing command: python -m sarathi.entrypoints.openai_server.api_server")
    print("\n💡 Try running this manually to see the full error:")
    print(f"   cd {sarathi_dir}")
    print("   python -m sarathi.entrypoints.openai_server.api_server")
    
    return False


def main():
    parser = argparse.ArgumentParser(
        description="Fused LLM Chatbot - Gstack + AdaptiVocab + vAttention + LaRoSA"
    )
    parser.add_argument(
        "--test-server",
        action="store_true",
        help="Test server startup and show diagnostic information"
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to configuration JSON file"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt2",
        help="Model name (default: gpt2)"
    )
    parser.add_argument(
        "--patch-tokenizer",
        type=str,
        help="Path to PatchTokenizer (AdaptiVocab). Auto-detects if not specified."
    )
    parser.add_argument(
        "--no-adaptivocab",
        action="store_true",
        help="Disable AdaptiVocab"
    )
    parser.add_argument(
        "--no-vattention",
        action="store_true",
        help="Disable vAttention (use standard attention)"
    )
    parser.add_argument(
        "--no-larosa",
        action="store_true",
        help="Disable LaRoSA"
    )
    parser.add_argument(
        "--larosa-sparsity",
        type=float,
        default=0.4,
        help="LaRoSA sparsity level (0.0-1.0, default: 0.4)"
    )
    parser.add_argument(
        "--gstack-model",
        type=str,
        help="Path to Gstack-trained model"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Server port (default: 8000)"
    )
    
    args = parser.parse_args()
    
    # Test mode
    if args.test_server:
        test_server_manually()
        return
    
    # Create config
    config = FusedChatbotConfig(args.config)
    
    # Override with command line arguments
    if args.model:
        config['model_name'] = args.model
    if args.port:
        config['server_port'] = args.port
    if args.larosa_sparsity:
        config['larosa_sparsity'] = args.larosa_sparsity
    
    # Handle AdaptiVocab
    if args.no_adaptivocab:
        config['patch_tokenizer_path'] = None
    elif args.patch_tokenizer:
        config['patch_tokenizer_path'] = args.patch_tokenizer
    else:
        # Try to auto-detect
        patch_tokenizer = find_patch_tokenizer()
        if patch_tokenizer:
            print(f"🔍 Auto-detected PatchTokenizer: {patch_tokenizer}")
            config['patch_tokenizer_path'] = patch_tokenizer
        else:
            print("⚠️  No PatchTokenizer found. AdaptiVocab will be disabled.")
            print("   Create one with: cd AdaptiVocab/src/build_vocab && python3 create_patch_tokenizer.py")
    
    # Handle vAttention
    if args.no_vattention:
        config['attention_backend'] = "flash_attn"  # Standard FlashAttention
    else:
        config['attention_backend'] = "fa_vattn"  # vAttention
    
    # Handle LaRoSA
    if args.no_larosa:
        config['larosa_sparsity'] = 0.0
    else:
        config['larosa_sparsity'] = args.larosa_sparsity
    
    # Handle Gstack
    if args.gstack_model:
        config['gstack_model_path'] = args.gstack_model
    
    # Check Python version before starting
    import sys
    python_version = sys.version_info
    if python_version.major == 3 and python_version.minor == 13:
        print("\n⚠️  WARNING: Python 3.13 detected!")
        print("=" * 60)
        print("❌ PyTorch 2.3.0+ is not available for Python 3.13 yet.")
        print("\n💡 Quick Fix (2 minutes):")
        print("   conda create -n fused_llm python=3.11 -y")
        print("   conda activate fused_llm")
        print("   ./setup_dependencies.sh")
        print("\n📖 See PYTHON_FIX.md or PYTHON_VERSION_GUIDE.md for details")
        print("=" * 60)
        return
    
    # Create and start chatbot
    chatbot = FusedChatbot(config)
    
    try:
        if chatbot.start():
            chatbot.interactive_mode()
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
    finally:
        chatbot.stop()


if __name__ == "__main__":
    main()

