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

import time

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
    #
    repo_id="drbh/yamoe",
    revision="v1",
    layer_name="Yamoe",
    trust_remote_code=True,
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

questions = [
    ("What is the capital of France?", "Paris"),
    ("What planet is closest to the Sun?", "Mercury"),
    ("What is the chemical symbol for water?", "H2O"),
    ("How many continents are there? (Answer in digits)", "7"),
    ("What is the speed of light in km/s?", "299"),
    ("Who wrote Romeo and Juliet?", "Shakespeare"),
    ("What is the largest ocean on Earth?", "Pacific"),
    ("What is the square root of 144?", "12"),
    ("What element has atomic number 1?", "Hydrogen"),
    ("How many legs does a spider have?", "8"),
    ("How does the human immune system fight off infections?", "antibod"),
    ("Explain the water cycle and why it matters for ecosystems.", "evaporation"),
    ("Describe how a neural network learns from data.", "backpropagation"),
    (
        "Explain how photosynthesis works step by step and note the key pigment involved.",
        "chlorophyll",
    ),
    (
        "Describe the causes and consequences of World War I. What event triggered the war?",
        "assassination",
    ),
]

max_new_tokens = 512


def run_assessment(label):
    print(f"\n{'=' * 50}")
    print(f" {label}")
    print(f"{'=' * 50}")

    correct = 0
    total_tokens = 0
    total_time = 0.0

    for question, expected in questions:
        messages = [{"role": "user", "content": f"Answer in one sentence: {question}"}]
        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        ).to(model.device)
        prompt_len = inputs["input_ids"].shape[1]

        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            generated = model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False
            )
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        new_tokens = generated[0, prompt_len:]
        n_tokens = len(new_tokens)
        text = tokenizer.decode(new_tokens, skip_special_tokens=True)

        total_tokens += n_tokens
        total_time += elapsed

        match = expected.lower() in text.lower()
        correct += int(match)
        status = "PASS" if match else "FAIL"
        print(f"  [{status}] {question}")
        print(f"         -> {text.strip()}")

    print("=" * 50)
    print(f"Score: {correct}/{len(questions)}")
    print("=" * 50)


speed_prompt = "Write a essay on the history of artificial intelligence, covering key milestones and figures in the field."
speed_max_tokens = 512


def run_speed(label):
    messages = [{"role": "user", "content": speed_prompt}]
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    ).to(model.device)
    prompt_len = inputs["input_ids"].shape[1]

    torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.no_grad():
        generated = model.generate(
            **inputs, max_new_tokens=speed_max_tokens, do_sample=False
        )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    new_tokens = generated[0, prompt_len:]
    n_tokens = len(new_tokens)
    text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    print(text)

    print("=" * 50)
    print(
        f"{label}: {n_tokens} tokens in {elapsed:.2f}s ({n_tokens / elapsed:.1f} tok/s)"
    )
    print("=" * 50)


# Run without yamoe (baseline).
run_assessment("Baseline (no yamoe)")
run_speed("Baseline speed")

print("\n\n")

# Swap the MoE for the yamoe kernel.
with use_kernel_mapping(mapping):
    model = kernelize(model, mode=Mode.INFERENCE)

# Run with yamoe.
run_assessment("With yamoe")
run_speed("Yamoe speed")
