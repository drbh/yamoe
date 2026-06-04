# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "transformers>=4.55.0",
#   "accelerate",
#   "triton",
#   "numpy",
#   "kernels==0.14.0",
#   "torch==2.11.0",
# ]
#
# [[tool.uv.index]]
# name = "pytorch-cu128"
# url = "https://download.pytorch.org/whl/cu128"
# explicit = true
#
# [tool.uv.sources]
# torch = { index = "pytorch-cu128" }
# ///

from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Mxfp4Config
from kernels import (
    LayerRepository,
    LocalLayerRepository,
    Mode,
    kernelize,
    use_kernel_mapping,
)

num_layers = 24

# Load the yamoe layer
repo_root = Path(__file__).resolve().parent.parent
# layer = LocalLayerRepository(
    # repo_path=repo_root / "result",
layer = LayerRepository(
    repo_id="drbh/yamoe",
    revision=1,
    layer_name="Yamoe",
)
mapping = {"MegaBlocksMoeMLP": {"cuda": layer}}

# Init the model (dequantize mxfp4 -> dense bf16 experts for the kernel).
model = AutoModelForCausalLM.from_pretrained(
    "openai/gpt-oss-20b",
    dtype=torch.bfloat16,
    device_map="cuda",
    quantization_config=Mxfp4Config(dequantize=True),
    num_hidden_layers=num_layers,
)
model.eval()

tokenizer = AutoTokenizer.from_pretrained("openai/gpt-oss-20b")

messages = [
    {"role": "user", "content": "In one sentence, what is a mixture of experts?"}
]
inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt",
    return_dict=True,
).to(model.device)

# Swap the MoE for the yamoe kernel.
with use_kernel_mapping(mapping):
    model = kernelize(model, mode=Mode.INFERENCE)

with torch.no_grad():
    generated = model.generate(**inputs, max_new_tokens=64, do_sample=False)

# Strip the prompt and decode only the newly generated tokens.
new_tokens = generated[0, inputs["input_ids"].shape[1] :]
print(tokenizer.decode(new_tokens, skip_special_tokens=True))
