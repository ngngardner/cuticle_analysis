{ pkgs ? import ./nix { } }:

let
  pythonLibs = pkgs.python39.buildEnv.override {
    extraLibs = [ (import ./default.nix { inherit pkgs; }).cuticle ];
  };
in
pkgs.mkShell {
  packages = [
    pythonLibs
    pkgs.python39Packages.autopep8
    pkgs.python39Packages.pycodestyle
    pkgs.python39Packages.pylint
  ];
}