import argparse
import json
from typing import Dict, Any
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.cache_utils import DynamicCache
import numpy as np
from flask import Flask, request, jsonify, render_template_string

from models import get_model_class, blend_gist_key_values
from reuse_pipeline import tokenize_for_reuse, prefill_kv_cache

# Initialize Flask app
app = Flask(__name__)

# Global variables for model and tokenizer
tokenizer = None
model = None
device = None

MODEL_GENERATE_API_WARNING_STRING = """==== PLEASE READ ====
With transformers==4.57.1 (which is required by this project), model.generate() API is buggy:
It is not compatible with custom position_ids, and it will cause incorrect results.
See https://github.com/huggingface/transformers/issues/36510 for how to fix it.
==== PLEASE READ ====
"""

def initialize_model(model_name: str):
    """Initialize the model and tokenizer globally"""
    global tokenizer, model, device
    
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    _, model_class = get_model_class(model_name, "qkv")
    model = model_class.from_pretrained(
        model_name, 
        device_map="auto",
        trust_remote_code=True,
        local_files_only=True,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2"
    )
    device = model.device
    print("Model loaded successfully!")

@app.route('/')
def index():
    """Render the main interface"""
    html_template = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Model Inference</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; }
            .container { max-width: 800px; margin: auto; }
            textarea { width: 100%; height: 150px; padding: 10px; margin: 10px 0; }
            button { background-color: #4CAF50; color: white; padding: 10px 20px; border: none; cursor: pointer; }
            button:hover { background-color: #45a049; }
            .result { background-color: #f0f0f0; padding: 15px; margin: 10px 0; border-radius: 5px; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Model Inference Service</h1>
            <form id="inferenceForm">
                <label for="system_prompt">System Prompt (optional):</label>
                <textarea id="system_prompt" placeholder="Enter system prompt..."></textarea>
                
                <label for="context">Context/Documents:</label>
                <textarea id="context" placeholder="Enter context/documents..."></textarea>
                
                <label for="question">Question:</label>
                <textarea id="question" placeholder="Enter your question..." required></textarea>
                
                <button type="submit">Generate Response</button>
            </form>
            
            <div id="result" class="result" style="display:none;">
                <h3>Generated Response:</h3>
                <p id="generated_text"></p>
            </div>
        </div>

        <script>
            document.getElementById('inferenceForm').addEventListener('submit', async function(e) {
                e.preventDefault();
                
                const systemPrompt = document.getElementById('system_prompt').value;
                const context = document.getElementById('context').value;
                const question = document.getElementById('question').value;
                
                const response = await fetch('/generate', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        system_prompt: systemPrompt,
                        context: context,
                        question: question
                    })
                });
                
                const data = await response.json();
                
                if (data.error) {
                    alert('Error: ' + data.error);
                } else {
                    document.getElementById('generated_text').innerText = data.prediction;
                    document.getElementById('result').style.display = 'block';
                }
            });
        </script>
    </body>
    </html>
    """
    return render_template_string(html_template)

@app.route('/generate', methods=['POST'])
def generate_response():
    """Generate response from the model based on input"""
    try:
        data = request.json
        system_prompt = data.get('system_prompt', '')
        context = data.get('context', '')
        question = data.get('question', '')
        
        if not question:
            return jsonify({'error': 'Question is required'}), 400
            
        # Prepare documents list from context
        documents = [item.strip() for item in context.split('# #') if item.strip()]
        print(documents)
        
        # Process inputs similar to the original evaluation code
        with torch.inference_mode():
            # Pre-compute system prompt if provided
            if system_prompt:
                system_inputs = tokenize_for_reuse(tokenizer, [system_prompt], keep_bos=True, role='system').to(device)
                system_cache = prefill_kv_cache(model, system_inputs)
                system_length = system_cache.get_seq_length()
            else:
                system_cache = None
                system_length = 0
            
            # Pre-compute context if provided
            if documents:
                context_inputs = tokenize_for_reuse(tokenizer, documents, keep_bos=False, role='user').to(device)
                model.model.config._attn_implementation = "flex_attention"
                outputs, gist_mask, pos_ids = model.model.generate_gist(ratio=4, **context_inputs)
                pos_ids = pos_ids[:, -gist_mask.shape[1]:]
                context_cache, _ = blend_gist_key_values(
                    model.config, [outputs.past_key_values], [gist_mask], [pos_ids],
                    model.model.rotary_emb, system_length
                )
                context_length = context_inputs.attention_mask.sum().item()
                precompute_length = pos_ids.max().item() + 1
                assert precompute_length == system_length + context_length, \
                    f"Precompute position id mismatch: {precompute_length} != {system_length} + {context_length}"
            else:
                context_cache = None
                precompute_length = system_length
                context_length = 0
            
            # Combine caches if both system and context exist
            if system_cache is not None and context_cache is not None:
                for system_layer, context_layer in zip(system_cache.layers, context_cache.layers):
                    context_layer.keys = torch.cat([system_layer.keys, context_layer.keys], dim=-2)
                    context_layer.values = torch.cat([system_layer.values, context_layer.values], dim=-2)
                combined_cache = context_cache
            elif system_cache is not None:
                combined_cache = system_cache
            elif context_cache is not None:
                combined_cache = context_cache
            else:
                combined_cache = None
            
            # Prepare input for question
            input_ids = tokenize_for_reuse(
                tokenizer, [question], keep_bos=False, role='user', add_generation_prompt=True
            ).input_ids.to(device)
            query_length = input_ids.shape[1]
            original_length = query_length + precompute_length
            position_ids = torch.arange(precompute_length, original_length, dtype=torch.long, device=device)
            mock_gist_ids = torch.full((1, combined_cache.get_seq_length() if combined_cache else 0), 0, dtype=torch.long, device=device) if combined_cache else torch.empty((1, 0), dtype=torch.long, device=device)
            input_ids = torch.cat([mock_gist_ids, input_ids], dim=1) if combined_cache else input_ids
            attention_mask = torch.ones_like(input_ids)

            # Generate text
            model.model.config._attn_implementation = "flash_attention_2"
            generated_outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids.unsqueeze(0),
                past_key_values=combined_cache,
                max_new_tokens=512,  # Default max new tokens
                pad_token_id=tokenizer.eos_token_id,
                use_cache=True,
                use_gist=True,
                do_sample=False,
            )
            
            # Decode generated text (skip query part)
            generated_tokens = generated_outputs[0][input_ids.shape[1]:]
            generated_tokens = generated_tokens[:512]  # Limit output length
            pred = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            
            return jsonify({
                'prediction': pred,
                'status': 'success'
            })
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def main():
    parser = argparse.ArgumentParser(description="Run model as Flask service")
    parser.add_argument("--model", type=str, required=True, 
                       help="Model name or path (e.g., 'Qwen/Qwen2-7B-Instruct', 'mistralai/Mistral-7B-v0.1')")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host for Flask app")
    parser.add_argument("--port", type=int, default=5000, help="Port for Flask app")
    
    args = parser.parse_args()

    print(MODEL_GENERATE_API_WARNING_STRING)
    
    # Initialize model
    initialize_model(args.model)
    
    print(f"Starting Flask server on {args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=False)

if __name__ == "__main__":
    main()
