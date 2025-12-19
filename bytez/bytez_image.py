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

# choose noobai-v-pred-10-with-eq-vae-v01-usable-sdxl
# model = sdk.model("John6666/noobai-v-pred-10-with-eq-vae-v01-usable-sdxl") # worst model
# model = sdk.model("mountain08/save") # best model HD
model = sdk.model("John6666/mumix-xl-v20-sdxl")


# send input to model
output = model.run("Juice WRLD")

print({"output": output })