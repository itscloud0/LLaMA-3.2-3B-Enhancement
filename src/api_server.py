from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Union
from llm_wrapper import LLMWrapper
import uvicorn

app = FastAPI(
    title="AI Assistant API",
    description="API for the AI Assistant powered by Llama 3.2-3B",
    version="1.0.0"
)

# Initialize the LLM wrapper
llm = LLMWrapper()

class GenerateRequest(BaseModel):
    prompt: str
    max_length: Optional[int] = 100
    temperature: Optional[float] = 0.7
    top_k: Optional[int] = 50
    num_samples: Optional[int] = 1
    use_few_shot: Optional[bool] = True
    use_chain_of_thought: Optional[bool] = True

class GenerateResponse(BaseModel):
    response: Union[str, List[str]]
    model: str = "meta-llama/Llama-3.2-3B"

@app.post("/generate", response_model=GenerateResponse)
async def generate_text(request: GenerateRequest):
    """
    Generate text from a prompt using the LLM.
    
    Args:
        request: GenerateRequest containing the prompt and generation parameters
        
    Returns:
        GenerateResponse containing the generated text
    """
    try:
        response = llm.generate_text(
            prompt=request.prompt,
            max_length=request.max_length,
            temperature=request.temperature,
            top_k=request.top_k,
            num_samples=request.num_samples,
            use_few_shot=request.use_few_shot,
            use_chain_of_thought=request.use_chain_of_thought
        )
        return GenerateResponse(response=response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "model": "meta-llama/Llama-3.2-3B"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 