{ pkgs ? import ./nix { } }:

let
  pythonLibs = pkgs.python39.buildEnv.override {
    extraLibs = [ (import ./default.nix { inherit pkgs; }).cuticle ];
  };
in
with pkgs.python39Packages;
pkgs.mkShell {
  packages = [
    pythonLibs

    # dev
    autopep8
    pycodestyle
    pylint
    pytest
    coverage
  ];
}
