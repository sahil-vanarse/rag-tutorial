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

# choose handler
model = sdk.model("walterheart/handler")

# send input to model
output = model.run("Hello I'm Sahil Vanarse, I am a software developer. I work as software engineer") # didnt worked

print({"output": output })