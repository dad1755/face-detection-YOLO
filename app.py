import os
import torch
from transformers import pipeline

# Set your Hugging Face API token if required
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"

# Model ID
model_id = "meta-llama/Llama-3.2-1B"

try:
    # Create the text generation pipeline
    pipe = pipeline(
        "text-generation", 
        model=model_id, 
        torch_dtype=torch.bfloat16, 
        device_map="auto"
    )

    # Generate text
    output = pipe("The key to life is")
    
    # Print the generated output
    print(output)

except Exception as e:
    print("Error:", str(e))
