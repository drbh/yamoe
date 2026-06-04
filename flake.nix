{
  description = "Flake for yamoe kernels";

  inputs = {
    kernel-builder.url = "path:/home/drbh/Projects/kernels";
  };

  outputs =
    {
      self,
      kernel-builder,
    }:
    kernel-builder.lib.genKernelFlakeOutputs {
      # `genKernelFlakeOutputs` derives the build rev from `self`, so it must be a flake
      # with a revision. ./dev-build.sh therefore builds its patched temp copy as
      # a throwaway git repo (not a bare `path:`) so `self.shortRev` resolves.
      inherit self;
      path = ./.;

      pythonCheckInputs =
        pkgs: with pkgs; [
          tqdm
          py-cpuinfo
          importlib-metadata
          torchmetrics
        ];
    };
}
