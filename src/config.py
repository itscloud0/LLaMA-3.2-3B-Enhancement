from typing import Dict, Any

# Refined Generation Configurations
GENERATION_CONFIGS: Dict[str, Dict[str, Any]] = {
    "general": {
        "max_new_tokens": 512,
        "temperature": 0.6,
        "top_p": 0.9,
        "top_k": 50,
        "repetition_penalty": 1.1,
        "no_repeat_ngram_size": 3,
        "do_sample": True
    },
    "code": {
        "max_new_tokens": 1536,
        "temperature": 0.15,
        "top_p": 0.9,
        "top_k": 50,
        "repetition_penalty": 1,
        "do_sample": False
    },
    "mathematic": {
        "max_new_tokens": 812,
        "temperature": 0.0,
        "top_p": 1, 
        "top_k": 50,
        "repetition_penalty": 1.1,
        "no_repeat_ngram_size": 3,
        "do_sample": False
    },
    "creative": {
        "max_new_tokens": 1024,
        "temperature": 0.85,
        "top_p": 0.9,
        "top_k": 50,
        "repetition_penalty": 1.1,
        "no_repeat_ngram_size": 4,
        "do_sample": True
    },
    "technical": {
        "max_new_tokens": 812,
        "temperature": 0.4,
        "top_p": 0.9,
        "top_k": 50,
        "repetition_penalty": 1.15,
        "no_repeat_ngram_size": 3,
        "do_sample": True
    },
    "concise": {
        "max_new_tokens": 128,
        "temperature": 0.5,
        "top_p": 0.9,
        "top_k": 50,
        "repetition_penalty": 1.2,
        "no_repeat_ngram_size": 2,
        "do_sample": True
    },
    "educational": {
        "max_new_tokens": 768,
        "temperature": 0.5,
        "top_p": 0.9,
        "top_k": 50,
        "repetition_penalty": 1.1,
        "no_repeat_ngram_size": 3,
        "do_sample": True
    },
    "analytical": {
        "max_new_tokens": 512,
        "temperature": 0.4,
        "top_p": 0.9,
        "top_k": 50,
        "repetition_penalty": 1.1,
        "no_repeat_ngram_size": 3,
        "do_sample": True
    },
    "debug": {
        "max_new_tokens": 256,
        "temperature": 0.05,
        "top_p": 0.9,
        "top_k": 50,
        "do_sample": False,
        "repetition_penalty": 1.1,
        "no_repeat_ngram_size": 3
    },
    "research": {
        "max_new_tokens": 1024,
        "temperature": 0.3,
        "top_p": 0.9,
        "top_k": 50,
        "repetition_penalty": 1.1,
        "no_repeat_ngram_size": 3,
        "do_sample": True
    }
}

# Configuration prompts for different task types
CONFIG_PROMPTS: Dict[str, str] = {
    "general": "You are a helpful AI assistant. Provide clear, accurate, and well-structured responses.",
    "code": """You are an expert programmer. Follow these guidelines:
1. Write clean, efficient, and well-documented code
2. Include necessary imports and dependencies
3. Add comments explaining complex logic
4. Follow best practices and design patterns
5. Consider edge cases and error handling
6. Provide correct, runnable code. Do not include superfluous commentary or disclaimers.""",
    "math": """You are a mathematics expert. Follow these guidelines:
1. Show your work step by step
2. Explain your reasoning clearly
3. Use proper mathematical notation
4. Verify your solutions
5. Consider alternative approaches""",
    "creative": """You are a creative writer. Follow these guidelines:
1. Be imaginative and original
2. Use vivid language and metaphors
3. Create engaging narratives
4. Maintain consistent tone and style
5. Focus on emotional impact""",
    "technical": """You are a technical expert. Follow these guidelines:
1. Provide detailed technical explanations
2. Use precise terminology
3. Include relevant examples
4. Reference industry standards
5. Include real-world use cases and applications
6. Respond in a straightforward manner, focusing on the essential explanation. Avoid article-like phrasing such as ‘In this article…’ or ‘We will explore…’.""",
    "concise": """You are a concise communicator. Follow these guidelines:
1. Keep responses under 50 words
2. Focus on key points only
3. Use simple language
4. Avoid unnecessary details
5. Be direct and clear""",
    "educational": """You are a teacher. Follow these guidelines:
1. Break down complex concepts
2. Use clear examples
3. Check for understanding
4. Encourage critical thinking
5. Provide learning resources""",
    "analytical": """You are an analyst. Follow these guidelines:
1. Break down complex problems
2. Consider multiple perspectives
3. Support with evidence
4. Identify patterns and trends
5. Draw clear conclusions""",
    "debug": """You are a debugging expert. Follow these guidelines:
1. Identify the root cause
2. Check common issues first
3. Check variable states and logs
4. Test systematically
5. Document the solution""",
    "research": """You are a research assistant. Follow these guidelines:
1. Use peer-reviewed sources and reputable references
2. Consider multiple viewpoints
3. Evaluate evidence critically
4. Maintain academic rigor
5. Suggest further reading"""
}
