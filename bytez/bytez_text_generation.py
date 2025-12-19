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

# choose phi-2
# model = sdk.model("microsoft/phi-2") best model
model = sdk.model("bigcode/starcoder2-3b") 

# send input to model
output = model.run("Tell me about the System Design Interview book.")

print({"output": output })