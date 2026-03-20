{
  description = "flake for testing pytorch";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs?ref=nixos-25.11";
  };

  outputs =
    { nixpkgs, ... }:
    let
      systems = [
        "x86_64-linux"
        "x86_64-darwin"
        "aarch64-linux"
        "aarch64-darwin"
      ];
      fpkgs = sys: nixpkgs.legacyPackages.${sys};
      merge =
        content:
        builtins.listToAttrs (
          map (sys: {
            name = sys;
            value = content sys;
          }) systems
        );
    in
    {
      devShell = merge (
        system:
        let
          pkgs = fpkgs system;
          cclib = pkgs.stdenv.cc.cc.lib;
        in
        pkgs.mkShell {
          shellHook = ''
            export LD_LIBRARY_PATH=${cclib}/lib:$LD_LIBRARY_PATH

            uv sync
            source .venv/bin/activate
          '';

          buildInputs =
            with pkgs;
            [
              nixfmt
              python3
              gcc
              uv
              cclib
              ruff
              python3Packages.pyqt6
              python3Packages.tensorboard
            ];
        }
      );
    };
}
