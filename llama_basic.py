import os

from dotenv import load_dotenv
from langchain.llms import LlamaCpp

# Load environment variables from .env file
load_dotenv()

# Access the local model path
model_path: str | None = os.getenv("LLM_MODEL_PATH")

print(f"Using local LLM model from: {model_path}")

# Load the model with speed optimizations
llm: object = LlamaCpp(
    model_path=model_path,
    temperature=1.0,  # Slightly higher for natural responses
    max_tokens=256,
    n_ctx=1024,  # Reduce context for speed if needed
    n_batch=1024,  # Increase batch size for faster execution
    n_gpu_layers=110,  # Offload to GPU
    verbose=False,  # Disable logging for speed
    top_k=50,  # Reduce search space
    top_p=0.7  # Limit randomness for quick responses
)

# Test the model
prompt: str = "Tell us a joke"
response: object = llm.invoke(prompt)
print(response)