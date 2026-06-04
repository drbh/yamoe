# /// script
# requires-python = "==3.10"
# dependencies = [
#   "torch==2.8.0",
#   "transformers>=4.55.0",
#   "accelerate",
#   "triton",
#   "numpy",
#   "kernels==0.13.0",
# ]
# [[tool.uv.index]]
# name = "pytorch-cu129"
# url = "https://download.pytorch.org/whl/cu129"
# explicit = true
# [tool.uv.sources]
# torch = [{ index = "pytorch-cu129" }]
# ///

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Mxfp4Config
from kernels import LayerRepository, Mode, kernelize, use_kernel_mapping

num_layers = 24

# Fetch the yamoe layer from the Hub and map it onto gpt-oss's MoE block.
layer = LayerRepository(
    repo_id="drbh/yamoe",
    layer_name="Yamoe",
    revision="main",
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
    yamoe_logits = model(**inputs).logits.float()

print(yamoe_logits)
