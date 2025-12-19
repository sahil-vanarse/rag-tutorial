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

# choose text-to-video-ms-1.7b
model = sdk.model("nachikethmurthy666/text-to-video-ms-1.7b")
# model = sdk.model("Lightricks/LTX-Video-0.9.7-dev") Didnt worked


# send input to model
output = model.run("A cat in a wizard hat walking down the street")

print({"output": output })