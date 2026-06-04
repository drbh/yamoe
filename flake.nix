{
  description = "Flake for yamoe kernels";

  inputs = {
    kernel-builder.url = "github:huggingface/kernel-builder";
  };

  outputs =
    {
      self,
      kernel-builder,
    }:
    let
      # --- Fast local builds -------------------------------------------------
      #
      # By default this flake builds the full release matrix: every applicable
      # Torch/CUDA variant in `kernel-builder`'s version set, each compiled for
      # every CUDA capability in `build.toml`. That is what gets published to
      # the Hub and is intentionally broad.
      #
      # For local iteration that is wasteful: this machine only ever runs one
      # Torch build on one GPU. Setting `YAMOE_DEV=1` narrows the build down to
      # the single Torch/CUDA variant below, which is the dominant build-time
      # cost (each variant is a full compile). The default (unset) is untouched,
      # so releases keep the full matrix.
      #
      #   # full release matrix (unchanged):
      #   nix build .#bundle
      #
      #   # fast single-Torch-variant build (needs --impure for the env read):
      #   YAMOE_DEV=1 nix build --impure .#bundle
      #
      # Override the target if your local Torch differs:
      #   YAMOE_DEV=1 YAMOE_DEV_TORCH=2.7 YAMOE_DEV_CUDA=12.8 \
      #     nix build --impure .#bundle
      #
      # To ALSO narrow the per-variant GPU arch (sm) — `cuda-capabilities` lives
      # in build.toml and cannot be overridden through flake args here — use
      # `./dev-build.sh`, which builds from a throwaway git copy with a patched
      # build.toml so your working tree stays clean.
      #
      # NOTE: env reads make eval impure, hence `--impure`. When YAMOE_DEV is
      # unset the dev expressions below are never evaluated (Nix is lazy), so the
      # default build stays pure and reproducible.
      devBuild = builtins.getEnv "YAMOE_DEV" != "";

      envOr =
        name: default:
        let
          v = builtins.getEnv name;
        in
        if v != "" then v else default;

      # Local Torch target. Defaults match this repo's `build/torch28-cu129`
      # artifacts. Override with YAMOE_DEV_TORCH / YAMOE_DEV_CUDA.
      devTorch = envOr "YAMOE_DEV_TORCH" "2.8";
      devCuda = envOr "YAMOE_DEV_CUDA" "12.9";

      # Keep just the one Torch/CUDA variant that matches this machine.
      pickVariant = builtins.filter (
        v: (v.torchVersion or "") == devTorch && (v.cudaVersion or "") == devCuda
      );
    in
    kernel-builder.lib.genFlakeOutputs {
      # `genFlakeOutputs` derives the build rev from `self`, so it must be a flake
      # with a revision. ./dev-build.sh therefore builds its patched temp copy as
      # a throwaway git repo (not a bare `path:`) so `self.shortRev` resolves.
      inherit self;
      path = ./.;

      torchVersions = default: if devBuild then pickVariant default else default;

      pythonCheckInputs =
        pkgs: with pkgs; [
          tqdm
          py-cpuinfo
          importlib-metadata
          torchmetrics
        ];
    };
}
