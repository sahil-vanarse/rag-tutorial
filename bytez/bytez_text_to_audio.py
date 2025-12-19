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

# choose musicgen-stereo-large
# model = sdk.model("facebook/musicgen-stereo-large") # none worked
model = sdk.model("facebook/musicgen-melody-large") # none worked

# send input to model
output = model.run("Moody jazz music with saxophones")

print({"output": output })