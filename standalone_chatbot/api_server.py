#!/usr/bin/env python3
"""
REST API Server for Standalone Fused LLM Chatbot
Provides OpenAI-compatible API endpoints
"""

import argparse
import json
from typing import List, Dict, Optional
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

from standalone_chat import FusedLLMEngine


app = FastAPI(title="Standalone Fused LLM API")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global engine
engine: Optional[FusedLLMEngine] = None


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    model: str = "gpt2"
    temperature: float = 0.7
    max_tokens: int = 100


class ChatResponse(BaseModel):
    choices: List[Dict]
    usage: Dict


@app.on_event("startup")
async def startup_event():
    """Initialize engine on startup."""
    global engine
    if engine is None:
        raise RuntimeError("Engine not initialized. Use --model flag.")


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatRequest):
    """OpenAI-compatible chat completions endpoint."""
    global engine
    
    if engine is None:
        raise HTTPException(status_code=500, detail="Engine not initialized")
    
    # Convert messages to conversation history
    conversation_history = []
    user_message = None
    
    for msg in request.messages:
        if msg.role == "user":
            user_message = msg.content
        else:
            conversation_history.append({"role": msg.role, "content": msg.content})
    
    if user_message is None:
        raise HTTPException(status_code=400, detail="No user message found")
    
    # Generate response
    try:
        response = engine.chat(
            user_message,
            conversation_history=conversation_history if conversation_history else None,
        )
        
        return ChatResponse(
            choices=[
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": response,
                    },
                    "finish_reason": "stop",
                }
            ],
            usage={
                "prompt_tokens": len(engine.tokenizer.encode(user_message)),
                "completion_tokens": len(engine.tokenizer.encode(response)),
                "total_tokens": len(engine.tokenizer.encode(user_message + response)),
            },
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy", "engine_loaded": engine is not None}


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": "Standalone Fused LLM API",
        "version": "1.0.0",
        "endpoints": {
            "chat": "/v1/chat/completions",
            "health": "/health",
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Standalone Fused LLM API Server")
    parser.add_argument(
        "--model",
        type=str,
        default="gpt2",
        help="Model name (default: gpt2)",
    )
    parser.add_argument(
        "--patch-tokenizer",
        type=str,
        default=None,
        help="Path to AdaptiVocab PatchTokenizer .pkl file",
    )
    parser.add_argument(
        "--larosa-sparsity",
        type=float,
        default=0.0,
        help="LaRoSA sparsity level (0.0-1.0, default: 0.0)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to use (default: cpu)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind to (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind to (default: 8000)",
    )
    
    args = parser.parse_args()
    
    # Initialize engine
    global engine
    print("🚀 Initializing Fused LLM Engine...")
    engine = FusedLLMEngine(
        model_name=args.model,
        patch_tokenizer_path=args.patch_tokenizer,
        larosa_sparsity=args.larosa_sparsity,
        device=args.device,
    )
    
    print(f"\n✅ Server starting on http://{args.host}:{args.port}")
    print(f"📖 API docs: http://{args.host}:{args.port}/docs\n")
    
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()

