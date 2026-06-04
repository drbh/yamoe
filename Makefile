

dev:
	nix run github:huggingface/kernels#kernel-builder -- build --variant torch211-cxx11-cu126-x86_64-linux --max-jobs 4 --cores 4