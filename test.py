from llm_wrapper import LLMWrapper

wrapper = LLMWrapper()

# Normal (default behavior)
print(wrapper.generate_text("What is the capital of France?"))

# Force Chain-of-Thought manually
print(wrapper.generate_text("Solve 32 * 78", chain_of_thought=True, sample_n=5))

# Auto Chain-of-Thought (smart detector)
print(wrapper.generate_text("Explain the impact of AI on society."))
