from flask import Flask, request, jsonify
from flask_cors import CORS
from llm_wrapper import LLMWrapper
import time
import os
from typing import Dict, Any

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Initialize the LLM wrapper
llm = LLMWrapper()

@app.route('/api/generate', methods=['POST'])
def generate_text():
    """Generate text based on the provided prompt and task type."""
    try:
        # Get request data
        data = request.get_json()
        prompt = data.get('prompt')
        task_type = data.get('task_type', 'general')
        
        # Validate input
        if not prompt:
            return jsonify({
                'error': 'Prompt is required'
            }), 400
        
        # Generate response
        start_time = time.time()
        response = llm.generate_text(
            input_text=prompt,
            task_type=task_type
        )
        end_time = time.time()
        
        # Return response
        return jsonify({
            'response': response,
            'task_type': task_type,
            'processing_time': end_time - start_time
        })
        
    except Exception as e:
        return jsonify({
            'error': str(e)
        }), 500

@app.route('/api/task_types', methods=['GET'])
def get_task_types():
    """Get a list of available task types and their descriptions."""
    try:
        from config import CONFIG_PROMPTS
        
        task_types = {
            task_type: {
                'description': prompt.split('\n')[0]  # Get first line as description
            }
            for task_type, prompt in CONFIG_PROMPTS.items()
        }
        
        return jsonify(task_types)
        
    except Exception as e:
        return jsonify({
            'error': str(e)
        }), 500

if __name__ == '__main__':
    # Get port from environment variable or use default
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port) 