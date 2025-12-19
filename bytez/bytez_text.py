"""
  pip i bytez
"""

from bytez import Bytez
import os
from dotenv import load_dotenv
load_dotenv()
# load api key from .env
key = os.getenv("BYTEZ_API_KEY")
sdk = Bytez(key)

# choose Qwen3-8B-DND-Almost-Human-C
model = sdk.model("wls04/code_llama")

# send input to model
output = model.run([
  {
    "role": "user",
    "content": "Can we fuck ?"
  }
])


result = output.output
print(result["role"])
print("*" * 50)
print(result["content"])