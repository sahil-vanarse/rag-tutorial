"""
  pip i bytez
"""

from bytez import Bytez
import os
from dotenv import load_dotenv

load_dotenv()
key = os.getenv("BYTEZ_API_KEY")
sdk = Bytez(key)

# choose Soren-Qwen3-4B-Instruct-Finance-v1
model = sdk.model("Jackrong/Soren-Qwen3-4B-Instruct-Finance-v1")

# send input to model
output = model.run({
  "context": "My name is Sahil and I live in India",
  "question": "Where do I live?"
})

print({"output": output })