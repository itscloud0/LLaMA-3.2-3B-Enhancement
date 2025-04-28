from flask import Flask, request, jsonify
from flask_cors import CORS
from llm_wrapper import LLMWrapper
import time
import os

app = Flask(__name__)
CORS(app)

llm = LLMWrapper()

@app.route('/api/generate', methods=['POST'])
def generate_text():
    try:
        data = request.get_json()
        prompt = data.get('prompt')
        task_type = data.get('task_type', None)
        chain_of_thought = data.get('chain_of_thought', False)
        sample_n = int(data.get('sample_n', 1))

        if not prompt:
            return jsonify({'error': 'Prompt is required'}), 400

        start_time = time.time()
        response = llm.generate_text(
            input_text=prompt,
            task_type=task_type,
            chain_of_thought=chain_of_thought,
            sample_n=sample_n
        )
        end_time = time.time()

        return jsonify({
            'response': response,
            'task_type': task_type if task_type else "auto",
            'chain_of_thought': chain_of_thought,
            'sample_n': sample_n,
            'processing_time': end_time - start_time
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/task_types', methods=['GET'])
def get_task_types():
    try:
        from config import CONFIG_PROMPTS
        task_types = {
            task_type: {
                'description': prompt.split('\n')[0]
            }
            for task_type, prompt in CONFIG_PROMPTS.items()
        }
        return jsonify(task_types)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)